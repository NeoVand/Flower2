import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from flask_socketio import SocketIO, emit, disconnect
# from flask_jsglue import JSGlue
import mne
from scipy import signal
from sklearn.decomposition import FastICA, MiniBatchDictionaryLearning
import json
import moviepy.editor as e
import numpy as np
from pylsl import StreamInlet, resolve_stream


global data
data = {}
data['connected'] = False
data['processed'] = False
data['media_file_paths'] = []

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def close_clip(vidya_clip):
    try:
        vidya_clip.reader.close()
        del vidya_clip.reader
        if vidya_clip.audio is not None:
            vidya_clip.audio.reader.close_proc()
            del vidya_clip.audio
        del vidya_clip
    except Exception:
        pass

def jsonify(dic):
    return json.loads(json.dumps(dic,cls=NumpyEncoder))

def remove_xfix(strs):
  pre = os.path.commonprefix(strs)
  rev = [word[::-1] for word in strs]
  post = os.path.commonprefix(rev)
  return [word.replace(pre,'').replace(post[::-1],'') for word in strs]

def symmetric_conv(d):
    drp = np.roll(d,1,axis=1)
    drn = np.roll(d,-1,axis=1)
    return np.vstack([drn,d,drp])
def causal_conv(d):
    drn = np.roll(d,-1,axis=1)
    return np.vstack([drn,d])
def repeatfunc(func,num,mat):
    if num==1:
        return func(mat)
    else:
        return repeatfunc(func,num-1,func(mat))
   



# def polar_distance(theta_array):
#     length = theta_array.shape[0]
#     A = anp.tile(theta_array,(length,1))
#     mat = A.T-A
#     return 2.0*(1.0-anp.cos(mat))

# def optim_order(sigmat):
#     sig = anp.array(sigmat)
#     corc = 1.0-anp.corrcoef(sig)
#     def loss(thetas):
#         difs = polar_distance(thetas)-corc
#         mults = anp.multiply(difs,difs)
#         sums = anp.sum(mults)
#         out = anp.sqrt(sums)
#         return out
#     loss_gradient_fun = grad(loss)
#     thetas = anp.zeros(len(corc))
#     print("Initial loss:", loss(thetas))
#     for i in range(100):
#         thetas -= loss_gradient_fun(thetas) * 0.01
#         print("Trained loss:", loss(thetas))
#     # do something with thetas

def highpass_filter(y, sr, lcf, hcf, fs):
  filter_low_stop_freq = lcf  # Hz
  filter_low_pass_freq = lcf+0.1  # Hz
  filter_high_pass_freq = hcf
  filter_high_stop_freq = hcf+0.1
  filter_order = fs

  # High-pass filter
  nyquist_rate = sr / 2.
  desired = (0, 0, 1, 1, 0,0)
  bands = (0, filter_low_stop_freq, filter_low_pass_freq, filter_high_pass_freq,filter_high_stop_freq,nyquist_rate)
  filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

  # Apply high-pass filter
  filtered_signal = signal.filtfilt(filter_coefs, [1], y)
  return filtered_signal

EDF_UPLOAD_FOLDER = os.path.join('uploads','eeg')
MEDIA_UPLOAD_FOLDER = os.path.join('uploads','media')
# MEDIA_UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = {'edf','bdf'}

app = Flask(__name__)
# jsglue = JSGlue(app)
app.config['EDF_UPLOAD_FOLDER'] = EDF_UPLOAD_FOLDER
app.config['MEDIA_UPLOAD_FOLDER'] = MEDIA_UPLOAD_FOLDER
app.secret_key = 'blahblahblah'
socketio = SocketIO(app,pingInterval = 10000, pingTimeout= 5000)

@app.route('/')
def index():
    global data
    data['connected']=False
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global data
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('no file, empty')
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print('upload successful, file_name: ',file.filename)
            data['edf_file_path'] = os.path.join(app.config['EDF_UPLOAD_FOLDER'], file.filename)
            file.save(data['edf_file_path'])
            return render_template('select.html')
    return render_template('index.html')


@socketio.on('connect')
def connect():
    global data
    if not data['connected']:
        emit('edf_info', load_edf())
        data['connected']=True

##################################################################################################
##################################################################################################


@socketio.on('init_process')
def raw_process(dic):
    frame = dic['frame']
    print('req_frame: ',frame)
    depth = dic['depth']
    print('req_depth: ',depth)
    global data
    # if data['processed']:
    #     emit('init_data',data['out'])
    # else:
        
    emit('process_message','Filtering Data...')
    socketio.sleep(0)
    
    print('processing raw data ... ')
    data['signal_raw'] = np.stack(data['selected_channel_list'])
    data['signal_raw'] = data['signal_raw'][:,::data['resampling_rate']] 
    data['signal_raw'] = highpass_filter(data['signal_raw'],data['sampling_freq'],data['lcf'],data['hcf'],data['filter_size'])
    # optim_order(data['signal_raw'])
    num_samples = len(data['signal_raw'][0])
    num_channels = len(data['signal_raw'])
    
    emit('process_message','Standardizing Data...')
    socketio.sleep(0)

    data['signal_raw'] = data['signal_raw']-np.mean(data['signal_raw'],axis=0).reshape(1,num_samples)
    data['signal_raw'] = data['signal_raw'] - np.mean(data['signal_raw'], axis=1).reshape(num_channels,1)
    data['signal_raw'] = data['signal_raw']/np.std(data['signal_raw'],axis=0).reshape(1,num_samples)
    print('raw data shape: ', data['signal_raw'].shape)

    
    emit('process_message','Dimensionality Reduction...')
    socketio.sleep(0)

    model = FastICA(n_components=4, random_state=0)
    d = data['signal_raw']
    # ddd = repeatfunc(symmetric_conv,2,d)
    print("d-shape:",d.shape)
    # print("ddd-shape:",ddd.shape)

    


    # model = MiniBatchDictionaryLearning(n_components=4, alpha=0.1,
    #                                         n_iter=10, batch_size=200,
    #                                         random_state=0,positive_dict=True)
    data['model'] = model.fit(d.T)
    data['components'] = model.components_.T
    # data['components'] = data['components']-np.mean(data['components'],axis=0).reshape(1,4)
    # data['components'] = data['components']/np.std(data['components'],axis=0).reshape(1,4)
    data['components'] = data['components']-np.mean(data['components'])
    data['components'] = data['components']/np.std(data['components'])

    print('shape of components:',data['components'].shape)
    data['W'] = data['model'].transform(d.T)

    
    data['W'] = data['W']-np.mean(data['W'],axis=0).reshape(1,4)
    data['W'] = 0.5+ 0.5*data['W']/np.std(data['W'],axis=0).reshape(1,4)
    print('file processed. Data is ready to be served')
    
    emit('process_message','Sending Data...')
    socketio.sleep(0)

    # shift = 0
    # data['out'] = {'raw':data['signal_raw'][:,shift:shift+500].T,'flower':data['signal_interp'][:,shift:shift+500].T,'color':data['W'][shift:shift+500]}
    
    data['out'] = {'sampling_freq':data['sampling_freq'],\
        'raw':data['signal_raw'][:,frame-depth:frame].T,\
            'color':data['W'][frame-depth:frame,0:3],\
                'thickness':data['W'][frame-depth:frame,3],\
                    'full_size':len(data['signal_raw'][0]),\
                        'labels':data['selected_channel_labels'],\
                             'media_annotated':data['media_annotated'],\
                                 'components':data['components'].T}

    data['out'] = jsonify(data['out'])
    emit('init_data',data['out'])
    print('init batch sent, num_samples=',len(data['out']['raw']))
        # data['processed'] = True

@socketio.on('batch')
def send_batch(dic):
    frame = dic['frame']
    depth = dic['depth']
    data['out'] = {'sampling_freq':data['sampling_freq'],\
        'raw':data['signal_raw'][:,frame-depth:frame].T,\
            'color':data['W'][frame-depth:frame,0:3],\
                'thickness':data['W'][frame-depth:frame,3],\
                    'full_size':len(data['signal_raw'][0]),\
                        'labels':data['selected_channel_labels']}
    data['out'] = jsonify(data['out'])
    emit('batch_data',data['out'])
    print(f"{100*frame/len(data['signal_raw'].T)}% of the data is sent")
    data['processed'] = True

@socketio.on('cor')
def cor(dic):
    f1 = dic['f1']
    f2 = dic['f2']
    corr = signal.correlate(data['signal_raw'].T, (data['signal_raw'].T)[f1:f2], mode='same',method='fft')[:,2]
    # corr = signal.correlate(data['W'], data['W'][f1:f2], mode='same',method='fft')[:,2]
    corr = corr - np.mean(corr)
    corr = corr/np.std(corr)
    socketio.emit("correlation",jsonify({'data':corr}))

def load_edf():
    global data
    standard_channels = []
    f = mne.io.read_raw_edf(data['edf_file_path'])
    channel_labels = f.ch_names
    data['channel_labels'] = channel_labels

    sampling_freqs = f.info['sfreq']
    data['sampling_freqs'] = sampling_freqs

    data['raw_annotations'] = mne.read_annotations(data['edf_file_path'])

    data['raw_channel_list']=f.get_data()
    print('data shape: ',data['raw_channel_list'].shape)
    print('standardizing ...')
    preview = data['raw_channel_list'][:,:500]
    print('first mean shape:',np.mean(preview,axis=1))
    preview = (preview.T - np.mean(preview,axis=1)).T
    for i in range(len(preview)):
        std = np.std(preview[i,:])
        if  std>0:
            preview[i,:] = preview[i,:]/std

    # for channel in data['raw_channel_list'][:,:500]:
    #     standard = channel -  np.mean(channel)
    #     standard = standard/np.std(standard)
    #     standard_channels.append(standard)

    return jsonify({'channel_labels':channel_labels,'sampling_frequency':sampling_freqs,'preview':preview})


@app.route('/settings',  methods=['GET', 'POST'])
def settings():
    global data
    if request.method == 'POST':
        selected_channel_indices  = [int(box) for box in request.form]
        data['selected_channel_list'] = [data['raw_channel_list'][index] for index in selected_channel_indices]
        data['selected_channel_labels'] = remove_xfix([data['channel_labels'][index] for index in selected_channel_indices])
        data['sampling_freq'] = data['sampling_freqs']
        return render_template('settings.html',f=int(data['sampling_freq']))
    return redirect(request.url)

@app.route('/gui',  methods=['GET', 'POST'])
def gui():
    global data
    if request.method == 'POST':
        data['resampling_rate']=int(request.form['sf'])

        ################ process annotations ##################
        a = data['raw_annotations']
        if len(a)>0:
            try:
                data['media_annotations'] = []
                for annot in a:
                    onset = annot['onset']
                    onset_frame = np.round(data['sampling_freq']*onset/data['resampling_rate']).astype(np.int)
                    value = int(annot['description'].split('#')[1])
                    data['media_annotations'].append([onset_frame,value])
                print("first annotation:",a[0])
                print("last annotation:",a[-1])
                data['media_annotated'] = True
                        #interpolate annotations
                annot = np.array(data['media_annotations'])
                annotations=[]
                for i in range(len(annot)-1):
                    p1 = annot[i]
                    p2 = annot[i+1]
                    num = p2[0]-p1[0]
                    chunk = np.round(np.linspace(p1,p2,num,False)).astype(np.int)
                    annotations.append(chunk)
                data['media_annotations'] = np.vstack(annotations)
            except:
                data['media_annotated'] = False
        else:
            data['media_annotated'] = False

        data['sampling_freq']= data['sampling_freq']/data['resampling_rate']
        data['lcf']=float(request.form['lcf'])
        data['hcf']=float(request.form['hcf'])
        data['filter_size']=int(request.form['fs'])
        return render_template('gui.html')
    return redirect(request.url)


@app.route('/media', methods=['POST'])
def media():
    global data
    if request.method == 'POST':
        file = request.files['file']
        print('upload successful, file_name: ',file.filename)
        data['media_file_paths'].append(os.path.join(app.config['MEDIA_UPLOAD_FOLDER'], file.filename))
        file.save(data['media_file_paths'][-1])

        clip = e.VideoFileClip(data['media_file_paths'][-1])
        # print('video frame rate: ', FR)
        FR = clip.fps
        close_clip(clip)

        out = {'file_name':file.filename,'annotations':data['media_annotations'],'frame_rate':FR}
        socketio.emit('video',jsonify(out))
        return "video uploaded"

@app.route('/uploads/media/<filename>')
def send_file(filename):
    try:
        return send_from_directory(app.config['MEDIA_UPLOAD_FOLDER'], filename, as_attachment=True)
    except:
        pass
@socketio.on('rt_stop')
def rt_stop(msg):
    print(msg)
    global data
    data['rt_status']=False

@socketio.on('rt')
def realtime(msg):
    print(msg)
    depth = msg['depth']
    global data
    data['buffer']=[]
    data['counter'] = 0

    stream_name = 'medialab-EEG'
    streams = resolve_stream('type', 'EEG')

    try:
        for i in range (len(streams)):

            if (streams[i].name() == stream_name):
                index = i
                print ("NIC stream available")

        print ("Connecting to NIC stream... \n")
        inlet = StreamInlet(streams[index])
        data['rt_status'] = True


    except NameError:
        print ("Error: NIC stream not available\n\n\n")
        emit('rt_error',{'message':'EEG not available'})

    while data['rt_status']:
        sample, timestamp = inlet.pull_sample()
        data['buffer'].append(sample)
        data['counter'] += 1
        if len(data['buffer'])>=depth and data['counter'] >= 10:
            chunk = np.array(data['buffer']).T
            del data['buffer'][:data['counter']]
            data['counter'] = 0
            chunk = highpass_filter(chunk,data['sampling_freq'],data['lcf'],data['hcf'],data['filter_size'])
            chunk = chunk-np.mean(chunk,axis=0).reshape(1,depth)
            chunk = chunk - np.mean(chunk, axis=1).reshape(32,1)
            chunk = chunk/np.std(chunk,axis=0).reshape(1,depth)

            latent = data['model'].transform(chunk.T)
            latent = latent-np.mean(latent,axis=0).reshape(1,4)
            latent = 0.5+ 0.5*latent/np.std(latent,axis=0).reshape(1,4)
            out = {'raw':chunk.T,'color':latent[:,0:3],'thickness':latent[:,3]}
            out = jsonify(out)
            socketio.sleep(0)
            socketio.emit('rt_data',out)
    

if __name__ == "__main__":
    socketio.run(app, debug=True)