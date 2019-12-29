import datetime
import sys

def datetime2unix_time(t):
    unix_time = (t - datetime.datetime(1970, 1, 1)).total_seconds()
    return unix_time

def save_after_n_steps(saving_freq, save_path):
    """
    Callback calls model.save method after n_steps
    :param saving_freq: int
    :param saving_freq: str
    """
    def save_after_n_steps(_locals, _globals):
        if (_locals['self'].num_timesteps % saving_freq == 0 and _locals['self'].num_timesteps > 10):
            _locals['self'].save(save_path)
    return save_after_n_steps

def record(record_path, record_freq=1000):
    def record(_locals, _globals):
        if (_locals['self'].num_timesteps % record_freq != 0):
            return
        """
        Re create an env and record a video for one episode
        """
        model = _locals['self']
        obs = model.env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = model.env.step(action)
            if dones:
                break
        unix_time = int(datetime2unix_time(datetime.datetime.utcnow()))
        path = record_path + "%s.html" % unix_time
        model.env.observation_space.render(path=path)
    return  record

def populating_memory(_locals,_globals):
    model  = _locals['self']
    if model.num_timesteps < model.learning_starts:
        sys.stdout.write("\rPopulating the memory {}/{}...".format(model.num_timesteps, model.learning_starts))
        sys.stdout.flush()


from tqdm.auto import tqdm


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class progressbar_callback(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        def callback_progressbar(local_, global_):
            self.pbar.n = local_["self"].num_timesteps
            self.pbar.update(0)

        return callback_progressbar

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

def compose_callback(*callback_funcs):  # takes a list of functions, and returns the composed function.
    def _callback(_locals, _globals):
        continue_training = True
        for cb_func in callback_funcs:
            if cb_func(_locals, _globals) is False:  # as a callback can return None for legacy reasons.
                continue_training = False
        return continue_training

    return _callback