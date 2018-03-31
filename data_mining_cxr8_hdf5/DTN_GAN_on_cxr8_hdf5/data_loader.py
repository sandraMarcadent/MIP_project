from datetime import datetime
from helpers import *
from parameters import *


class Data_loader():
    """The CycleGAN module."""

    def __init__(self, pre_process=False, do_save=False, _data_path=DEFAULT_PATH,
                 _out_dir='./processed_data'):
        
        
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # parameters of control
        self.pre_process = pre_process
        self.do_save = do_save
        self.out_dir=_out_dir
        self.data_path = _data_path
        
        # num images to open/save
        self.max_images_to_open = MAX_IMAGES
        self.max_images_to_save = SAVE_MAX_IMAGES
        
        # image pre-processing parameters
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH
        self.img_channels = IMG_CHANNELS
        self.img_size = IMG_HEIGHT*IMG_WIDTH
        self.normalize = True
        
        # data processing parameters
        self.do_split = True
        self.prop_test = 0.1
        self.prop_val = 0.1
        
        self.correct_path = (self.out_dir+'/num_images='+ str(self.max_images_to_save)
                                                     +'/img_heigh=('+str(self.img_height)+
                                                    ')_img_width=('+str(self.img_width)+')'+
                                                    '_img_channels=('+str(self.img_channels)+')')
        
    
    def check_path(self):
        

        if os.path.exists(self.correct_path+'/data.h5'):

                    print('WARNING: data type already exists!!')
                    print('WARNING: to avoid time consumption, change path to true path...')
                    self.data_path = self.correct_path+'/data.h5'

                    if self.pre_process:
                        print('WARNING: nothing to preprocess, data already exists...')
                        self.pre_process=False

                    if self.do_save:                 
                        print('WARNING: nothing to save, data already exists...')
                        self.do_save=False

    

    def load_data(self):
        
        if self.do_save: self.max_images_to_open = self.max_images_to_save
    
        with h5py.File(self.data_path, 'r') as hdf:
            
            if (self.max_images_to_open > 0):
                X = hdf['X'][:self.max_images_to_open]
                _y = hdf['y'][:self.max_images_to_open]
            else:
                X = hdf['X'][:]
                _y = hdf['y'][:]

            if X.ndim < 4: 
                self.unprocessed_X = np.expand_dims(X, axis=3)
            else:
                self.unprocessed_X = X
                
            self.y = _y
            self.unprocessed_size = (self.unprocessed_X.shape[1]*
                                     self.unprocessed_X.shape[2])
            
            ###### create pseudo processed template
            self.processed_X = self.unprocessed_X
    

    def pre_process_data(self):
        
        # check for big data case
        self.total_num_images = self.unprocessed_X.shape[0]
        big_data = (self.total_num_images > 1000)
        
        # processing conditions
        resize = self.img_size != self.unprocessed_size
        rescale_grays = self.normalize
        convert_to_rgb = (self.img_channels == 3) & (self.unprocessed_X.shape[3] != 3)
        
        
        if big_data:

                print('WARNING: processing big data will be time consuming')
                print('total number of images to treat :', self.total_num_images)
    
                batch_list = []
                n_batches = int(self.total_num_images/1000)
    
                for i in range(0, n_batches):
                    
                    batch_i = self.processed_X[i*1000:(i+1)*1000]
                    batch_i = pre_processing_manager(batch_i, self.img_height, self.img_width,
                                                     resize=resize, rescale_grays=rescale_grays,
                                                     convert_to_rgb=convert_to_rgb)
                    batch_list.append(batch_i)
                    
                    if (i==n_batches-1) & ((i+1)*1000 < self.total_num_images): 
                        
                        last_batch = self.processed_X[(i+1)*1000:]
                        last_batch = pre_processing_manager(last_batch, self.img_height, self.img_width,
                                                     resize=resize, rescale_grays=rescale_grays,
                                                     convert_to_rgb=convert_to_rgb)
                        batch_list.append(last_batch)
                        
                self.processed_X = np.concatenate(batch_list, axis=0)
                
        else:
                
                self.processed_X = pre_processing_manager(self.processed_X, self.img_height, self.img_width,
                                                     resize=resize, rescale_grays=rescale_grays,
                                                     convert_to_rgb=convert_to_rgb)
                
    
    def format_to_classes(self):                      
       
           self.processed_a = self.processed_X[np.where(self.y==1)]
           self.processed_b = self.processed_X[np.where(self.y==0)]
        
    
    def split_data(self): 
            
            # CLASS A SPLIT
            lim_test = int(self.processed_a.shape[0]*self.prop_test)
            lim_val = lim_test+ int(self.processed_a.shape[0]*(self.prop_val))

            self.a_train = self.processed_a[:-lim_val,:,:]
            self.a_val = self.processed_a[-lim_val:-lim_test,:,:]
            self.a_test = self.processed_a[-lim_test:,:,:]
            
            # CLASS B SPLIT
            lim_test = int(self.processed_b.shape[0]*self.prop_test)
            lim_val = lim_test+ int(self.processed_b.shape[0]*(self.prop_val))
            
            self.b_train = self.processed_b[:-lim_val,:,:]
            self.b_val = self.processed_b[-lim_val:-lim_test,:,:]
            self.b_test = self.processed_b[-lim_test:,:,:]
            
            #FULL DATA SPLIT
            
            lim_test = int(self.processed_X.shape[0]*self.prop_test)
            lim_val = lim_test+ int(self.processed_X.shape[0]*(self.prop_val))
            
            self.x_train = self.processed_X[:-lim_val,:,:]
            self.x_val = self.processed_X[-lim_val:-lim_test,:,:]
            self.x_test = self.processed_X[-lim_test:,:,:]
            
            self.y_train = self.y[:-lim_val]
            self.y_val = self.y[-lim_val:-lim_test]
            self.y_test = self.y[-lim_test:]

            
            
    def  build_data(self):
        
            print('specified target data loader: ', (self.max_images_to_open,
                                                 self.img_height,
                                                 self.img_width,
                                                 self.img_channels))
            
            if self.do_save:
                print('specified target data saver: ', (self.max_images_to_save,
                                                     self.img_height,
                                                     self.img_width,
                                                     self.img_channels))
                
            
            #Check if preprocessing is necessary
            if (self.pre_process) or (self.do_save):
                            print('checking data path...')
                            self.check_path()
                             
            
            #Load data
            self.load_data()
            
            # Data Info
            if not self.pre_process:
                print('WARNING: you are not going to preprocess')
                print('check images properties')
                print('data size is : ', self.processed_X.shape)
 

            if self.pre_process:
                            print('processing data...')
                            self.pre_process_data()
                        
            if self.do_save:
                            print('saving processed data...')
                            
                            os.makedirs(self.correct_path)
                            
                            h5f = h5py.File(self.correct_path+'/data.h5', 'w')
   
                            h5f.create_dataset('X', data=self.processed_X)
                            h5f.create_dataset('y', data=self.y)
                            h5f.close()
        
            # Always format to classes
            self.format_to_classes()
                             
            if self.do_split:
                             self.split_data()