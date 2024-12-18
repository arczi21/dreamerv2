import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from scipy.stats import norm

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common
from common import NormLayer
from plugen.features import get_features
from plugen.net import NiceFlow


def main():

  configs = yaml.safe_load((
      pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  import tensorflow as tf
  from tensorflow.keras import layers as tfkl
  from tensorflow_probability import distributions as tfd
  from tensorflow.keras.mixed_precision import experimental as prec

  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
      capacity=config.replay.capacity // 10,
      minlen=config.dataset.length,
      maxlen=config.dataset.length))
  step = common.Counter(train_replay.stats['total_steps'])
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(config.eval_every)
  should_expl = common.Until(config.expl_until // config.action_repeat)

  def make_env(mode):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
      env = common.DMC(
          task, config.action_repeat, config.render_size, config.dmc_camera)
      env = common.NormalizeAction(env)
    elif suite == 'atari':
      env = common.Atari(
          task, config.action_repeat, config.render_size,
          config.atari_grayscale)
      env = common.OneHotAction(env)
    elif suite == 'crafter':
      assert config.action_repeat == 1
      outdir = logdir / 'crafter' if mode == 'train' else None
      reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
      env = common.Crafter(outdir, reward)
      env = common.OneHotAction(env)
    else:
      raise NotImplementedError(suite)
    env = common.TimeLimit(env, config.time_limit)
    return env

  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in config.log_keys_video:
        logger.video(f'{mode}_policy_{key}', ep[key])
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()

  print('Create envs.')
  num_eval_envs = min(config.envs, config.eval_eps)
  if config.envs_parallel == 'none':
    train_envs = [make_env('train') for _ in range(config.envs)]
    eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
  else:
    make_async_env = lambda mode: common.Async(
        functools.partial(make_env, mode), config.envs_parallel)
    train_envs = [make_async_env('train') for _ in range(config.envs)]
    eval_envs = [make_async_env('eval') for _ in range(eval_envs)]
  act_space = train_envs[0].act_space
  obs_space = train_envs[0].obs_space
  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  prefill = max(0, config.prefill - train_replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(act_space)
    train_driver(random_agent, steps=prefill, episodes=1)
    eval_driver(random_agent, episodes=1)
    train_driver.reset()
    eval_driver.reset()

  print('Create agent.')
  train_dataset = iter(train_replay.dataset(**config.dataset))
  report_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, obs_space, act_space, step)
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(train_dataset))
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      train_agent(next(train_dataset))
  train_policy = lambda *args: agnt.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  def train_step(tran, worker):
    if should_train(step):
      for _ in range(config.train_steps):
        mets = train_agent(next(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agnt.report(next(report_dataset)), prefix='train')
      logger.write(fps=True)
  train_driver.on_step(train_step)

  import matplotlib.pyplot as plt

  def get_trajectory(data, h=15):
    embed = agnt.wm.encoder(data)
    post, prior = agnt.wm.rssm.observe(embed, data['action'], data['is_first'])
    state = {k: v[0] for k, v in post.items()}
    states = [state]

    for i in range(h - 1):
      feat = agnt.wm.rssm.get_feat(state)
      dist = agnt._task_behavior.actor(feat)
      action = dist.sample()
      state = agnt.wm.rssm.img_step(state, action)
      states.append(state)

    return states

  def show_grid(images, grid, title=""):
    img_grid = images.reshape((grid[0], grid[1]) + images.shape[1:])
    img_grid = np.column_stack(np.column_stack(img_grid))
    plt.title(title)
    plt.imshow(img_grid, cmap='gray')
    plt.show()

  def get_image(state):
    feat = agnt.wm.rssm.get_feat(state)
    img = agnt.wm.heads['decoder'](feat)['image'].mode() + 0.5
    img = img.numpy()
    return img

  def save_model(path, model, optimizer):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


  def load_model(path, model, optimizer):
    loaded_state = torch.load(path)
    model.load_state_dict(loaded_state["model"])
    optimizer.load_state_dict(loaded_state["optimizer"])
    model.eval()
    return model, optimizer


  dataset = iter(train_replay.dataset(1, 1))

  lr = 1e-4
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  flow = NiceFlow(input_dim=1536, n_layers=4, n_couplings=4,
                  hidden_dim=512).to(device)
  optimizer = optim.Adam(flow.parameters(), lr=lr)

  flow, optimizer = load_model("plugen.pch", flow, optimizer)

  """

  data = next(dataset)

  features = torch.tensor([[[1., 1.]]]).to(device)
  current_mean = 2 * features - 1.
  current_sigma = torch.ones_like(current_mean) * 1.
  images = []

  for i in range(4):

    c = torch.normal(current_mean, current_sigma)
    s = torch.normal(torch.zeros(1, 1, 1534), torch.ones(1, 1, 1534))
    l = torch.cat((c, s), 2).to(device)
    with torch.no_grad():
      embed = flow.inv_flow(l)

    embed = tf.convert_to_tensor(embed.numpy()[0])
    embed = tf.convert_to_tensor(embed.numpy(), dtype=tf.dtypes.float16)
    deter = agnt.wm.rssm.initial(1)['deter']

    x = tf.concat([deter, embed], -1)
    x = agnt.wm.rssm.get('obs_out', tfkl.Dense, agnt.wm.rssm._hidden)(x)
    x = agnt.wm.rssm.get('obs_out_norm', NormLayer, agnt.wm.rssm._norm)(x)
    x = agnt.wm.rssm._act(x)
    stats = agnt.wm.rssm._suff_stats_layer('obs_dist', x)
    dist = agnt.wm.rssm.get_dist(stats)
    stoch = dist.sample()
    state = {'stoch': stoch, 'deter': deter, **stats}

    #img = np.array([get_image(post)[0]])
    #show_grid(img, (1, 1))
    feat = agnt.wm.rssm.get_feat(state)
    img = agnt.wm.heads['decoder'](feat)['image'].mode() + 0.5
    images.append(img[0].numpy())

    H = 15
    for j in range(H):
      dist = agnt._task_behavior.actor(feat)
      action = dist.sample()
      state = agnt.wm.rssm.img_step(state, action)
      feat = agnt.wm.rssm.get_feat(state)
      img = agnt.wm.heads['decoder'](feat)['image'].mode() + 0.5
      images.append(img[0].numpy())

  print(len(images))
  images = np.array(images).clip(0, 0.558)
  show_grid(images, (4, 16))
  
  """

  def replace_attributes(replay, flow, feature_idx, feature_vals, N=6, expand=False, expand_len=10, title=""):
    images = []
    for i in range(N):
      data = next(iter(replay.dataset(1, 1)))
      data = agnt.wm.preprocess(data)
      images.append(data['image'][0, 0].numpy() + 0.5)
      embed = agnt.wm.encoder(data).numpy()
      embed = torch.tensor(embed).to(device)
      with torch.no_grad():
        z, logdet = flow(embed)
      for idx, val in zip(feature_idx, feature_vals):
        z[:, :, idx] = val
      with torch.no_grad():
        embed = flow.inv_flow(z)
      embed = tf.convert_to_tensor(embed.numpy()[0], dtype=tf.dtypes.float16)
      deter = agnt.wm.rssm.initial(1)['deter']

      x = tf.concat([deter, embed], -1)
      x = agnt.wm.rssm.get('obs_out', tfkl.Dense, agnt.wm.rssm._hidden)(x)
      x = agnt.wm.rssm.get('obs_out_norm', NormLayer, agnt.wm.rssm._norm)(x)
      x = agnt.wm.rssm._act(x)
      stats = agnt.wm.rssm._suff_stats_layer('obs_dist', x)
      dist = agnt.wm.rssm.get_dist(stats)
      stoch = dist.sample()
      state = {'stoch': stoch, 'deter': deter, **stats}

      feat = agnt.wm.rssm.get_feat(state)
      img = agnt.wm.heads['decoder'](feat)['image'].mode() + 0.5
      images.append(img[0].numpy())

    images = np.array(images)
    print(images.shape)
    show_grid(images, (N, 2), title=title)

  def feature_histogram(replay, flow, feature_idx, feature_value, N=1024, title=""):
    count = 0
    values = []
    while count < N:
      data = next(iter(replay.dataset(1, 1)))
      features = get_features(data).reshape(-1)
      if features[feature_idx] == feature_value:
        count += 1
        print("%s / %s" % (count, N))
        data = agnt.wm.preprocess(data)
        embed = agnt.wm.encoder(data).numpy()
        embed = torch.tensor(embed).to(device)
        with torch.no_grad():
          z, logdet = flow(embed)
        values.append(z[0, 0, feature_idx])

    plt.hist(values, bins=np.linspace(2*feature_value-3, 2*feature_value + 1, 30), density=True)
    xmin, xmax = plt.xlim()
    x_range = np.linspace(xmin, xmax, 100)
    plt.plot(x_range, norm.pdf(x_range, loc=2*feature_value-1, scale=1), color='red')
    plt.title(title)
    plt.show()


  replace_attributes(train_replay, flow, [2], [1], N=8, title="time = 0")
  replace_attributes(train_replay, flow, [2], [0], N=8, title="time = 1")
  #replace_attributes(train_replay, flow, [0., 1.], N=8, title="RIGHT DOWN")
  #replace_attributes(train_replay, flow, [0., 0.], N=8, title="RIGHT UP")

  #feature_histogram(train_replay, flow, 0, 1, title="LEFT")
  #feature_histogram(train_replay, flow, 0, 0, title="RIGHT")
  #feature_histogram(train_replay, flow, 1, 1, title="DOWN")
  #feature_histogram(train_replay, flow, 1, 0, title="UP")


if __name__ == '__main__':
  main()
