"""

Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""
import json
import joblib
import shutil
import numpy as np
import torch
import os.path as osp, time, atexit, os
from utils.mpi_tools import proc_id, mpi_statistics_scalar
from utils.serialization_utils import convert_json

import moviepy.editor as mpy
import cv2


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file
                containing metrics logged throughout a training run.
                Defaults to ``progress.txt``.

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        
        if proc_id()==0:
            self.output_dir = output_dir or "/tmp/experiments/%i"%int(time.time())
            if osp.exists(self.output_dir):
                print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)
            print(colorize("Logging data to %s"%self.output_file.name, 'green', bold=True))
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name
        self.vis = None
        
    def add_vis(self, vis):
        self.vis = vis


    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        if proc_id()==0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        del_key = ['vis', 'out_draw', 'temp_out_draw', 'out', 'out_j', 'out_c', 'env']
        for i in del_key:
            if i in config.keys():
                del config[i]
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if proc_id()==0:
            output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
            print(colorize('Saving config:\n', color='cyan', bold=True))
            print(output)
            with open(osp.join(self.output_dir, "config.json"), 'w') as out:
                out.write(output)

    def save_state(self, state_dict, models, itr=None):
        """
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent copy of the model via ``model``.

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.
            model (nn.Module): A model which contains the policy.
            itr: An int, or None. Current iteration of training.
        """
        if proc_id()==0:
            # fname = 'vars.pkl' if itr is None else 'vars.%d.pkl'%itr
            # try:
            #     joblib.dump(state_dict, osp.join(self.output_dir, fname))
            # except:
            #     self.log('Warning: could not pickle state_dict.', color='red')
            m = models[0]
            torch.save(m['sas'], osp.join(self.output_dir, 'sas'+('.pt' if itr is None else '.%d.pt'%itr) ))
            torch.save(m['sa'], osp.join(self.output_dir, 'sa'+('.pt' if itr is None else '.%d.pt'%itr) ))
            for m in models[1:]:
                self._torch_save(m, itr)
    
    def save_agent(self, agent_dict):
        if proc_id()==0:
            fname = 'agent.pkl'
            try:
                joblib.dump(agent_dict, osp.join(self.output_dir, fname))
            except:
                self.log('Warning: could not pickle agent_dict.', color='red')
                

    def _torch_save(self, model, itr=None):
        if proc_id()==0:
            fname = model.__class__.__name__ + ('.pt' if itr is None else '.%d.pt'%itr)
            torch.save(model, osp.join(self.output_dir, fname))

    def dump_tabular(self, isprint=True, isvis=True):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        if proc_id()==0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15,max(key_lens))
            keystr = '%'+'%d'%max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-"*n_slashes) if isprint else None
            for key in self.log_headers:    
                val = self.log_current_row.get(key, "")
                if key == 'Z'  and self.vis is not None and isvis:
                    valmean = val[0][:20]
                    valindex = val[2][:20]
                    self.vis.scatter(
                            X=valmean,
                            Y=valindex+1,
                            win=key,
                            opts=dict(title=key,xtickmin=-2.5,xtickmax=2.5,ytickmin=-2.5,ytickmax=2.5),
#                            update='new'
                            )
                    continue
                if key == 'Map' and self.vis is not None and isvis:
                    if not len(val.shape) == 3:
                        self.vis.heatmap(X = val, win=key, opts=dict(title=key))
                        continue 
                    index = 0
                    self.vis.heatmap(X = val.sum(0), win=key, opts=dict(title=key))
                    for m in val:
                        index += 1
                        self.vis.heatmap(X = m, \
                                         win=key+'-'+str(index), \
                                         opts=dict(title=key+'-'+str(index)),)
                    continue
                valstr = "%8.3g"%val if hasattr(val, "__float__") else val
                print(fmt%(key, valstr)) if isprint else None
                if self.vis is not None and isvis:
                    self.vis.line(
                            X=np.array([self.log_current_row.get('Epoch', "")]),
                            Y=np.array([val]),
                            win=key,
                            opts=dict(title=key),
                            update='append'
                            )
                vals.append(val)
            print("-"*n_slashes) if isprint else None
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers)+"\n")
                self.output_file.write("\t".join(map(str,vals))+"\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False
        
        
    def save_video(self, visual_obs, video_name, fpath, vr='video'):
        record_dir = osp.join(self.output_dir if fpath is None else fpath, vr)
        if not osp.exists(record_dir):
            os.makedirs(record_dir)
            
        video_path = osp.join(record_dir, video_name)
        fps = 10. #128
        
        def f(t):
            frame_length = len(visual_obs)
            new_fps = 1./(1./fps + 1./frame_length)
            idx = min(int(t*new_fps), frame_length-1)
            # return visual_obs[idx]
            return cv2.cvtColor(visual_obs[idx], cv2.COLOR_BGR2RGB)
        video = mpy.VideoClip(f, duration=len(visual_obs)/fps+2)
        # video =video.resize(height=800)
        video.write_videofile(video_path, fps, verbose=False,)# progress_bar=False)# LJX CHANGED
        
        cv2.imwrite(osp.join(record_dir, 'didi.jpg'), visual_obs[-1]) #cv2.cvtColor(visual_obs[-1], cv2.COLOR_BGR2RGB)
        print(colorize("Video has been saved, dir = " + video_path, 'green', bold=True))
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path, audio_fps=10)
        clip.write_gif(video_path[:-4]+'.gif', fps=10)
        # clip = mpy.VideoFileClip(states_path)#.resize((450, 195))
        # clip.write_gif(states_path[:-4]+".gif", fps = 8)
        
        
#     def load_video(self, info, fpath):
#         ci, max_con, sm, sr, con, ckpt, seed = info
        
#         sub = 'v-'+str(sm)+'-'+str(sr)+'-'+str(ci)+'-'+str(max_con)+'-'+str(seed)
# #        fpath = osp.join(fpath, os.path.pardir, sub, 'video')
#         fpath = osp.join(fpath, 'video') if ci==-1 else osp.join(fpath, os.path.pardir, sub, 'video')
#         f = 'Con'+'-'+str(con)+'-'+str(ckpt)+'-'+str(seed)
#         print(fpath)
#         video = []
#         videoCapture = cv2.VideoCapture(osp.join(fpath, f+'.mp4'))
#         fps = videoCapture.get(cv2.CAP_PROP_FPS)
#         size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
#                 int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
#         success, frame = videoCapture.read()
#         video.append(frame)
#         while success:
# #                cv2.imshow(f, frame)
# #                cv2.waitKey(1000/int(fps))
#             success, frame = videoCapture.read()
#             video.append(frame)
#         return video, size

        
#        (args.env_name, str(args.skill_mode), str(args.skill_rep), str(args.con), str(ckpt_num), str(args.seed))
    def save_reps(self, records, fpath):
        fpath = osp.join(fpath, os.path.pardir)
        
        info = ''
        for i in records['info'][1:]:
            info = info + i + '-'
        info = ['Record', info[:-1]]
        self.save_video(records['video'], info, fpath, 'records')
        
        states_path = osp.join(fpath, 'records', info[1]+'.npy')
        np.save(states_path, np.array(records['states']))
        print(colorize("States has been saved, dir = " + states_path, 'green', bold=True))
        
        
        
    def load_reps(self, info, fpath):
        sr_, max_con, sm, sr, con, ckpt, seed = info
        f = str(sm)+'-'+str(sr)+'-'+str(con)+'-'+str(ckpt)+'-'+str(seed)
        fpath = osp.join(fpath, os.path.pardir, 'records')
        if sr_ in [1, 2]:
            print((colorize("Leading states, dir = " + osp.join(fpath, f+'.npy'), 'green', bold=True)))
            states = np.load(osp.join(fpath, f+'.npy'))
            size = states.shape
            return states, size
        elif sr_ in [3, 4]:
            video = []
            print((colorize("Leading video, dir = " + osp.join(fpath, f+'.mp4'), 'green', bold=True)))
            videoCapture = cv2.VideoCapture(osp.join(fpath, f+'.mp4'))
            fps = videoCapture.get(cv2.CAP_PROP_FPS)
            size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            success, frame = videoCapture.read()
            video.append(frame)
            while success:
#                cv2.imshow(f, frame)
#                cv2.waitKey(1000/int(fps))
                success, frame = videoCapture.read()
                video.append(frame)
            return video, size
        else:
            pass
        return None
        
    

class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()
        

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k,v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            if isinstance(v, list):
                self.epoch_dict[k] += v
            else:
                self.epoch_dict[k].append(v)
                

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key,val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key, stats[0])
            if not(average_only):
                super().log_tabular('Std'+key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max'+key, stats[3])
                super().log_tabular('Min'+key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        return mpi_statistics_scalar(vals)