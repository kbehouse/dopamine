from segnet_label import SegnetLabel
# Tompson Sampling (TS) with multiprocessing
from multiprocessing import Process, Value, Lock, Manager, Pipe
import sys, os
import time
import cv2 

from shutil import copyfile

#----------------_NOT Use this, only for test-------------- use semantic_process#

def process_func(child_conn, d):
    print('-----------process_func __file__ = ', __file__)
    # s = SegnetLabel( n_classes=2, input_height=224, input_width=224, save_weights_path='weights/ex1', epoch_number=10 )
    s = SegnetLabel( n_classes=4, input_height=224, input_width=224, save_weights_path='weights/3obj', epoch_number=5 )
     
    suffix = time.strftime("%y%b%d_%H%M%S")
    predict_dir = 'semantic_annotate_22' + suffix
    predict_dir  = os.path.abspath(predict_dir)
    if os.path.exists(predict_dir):
        import shutil
        shutil.rmtree(predict_dir)
    print("Make dir: " + predict_dir)
    os.makedirs(predict_dir)
    
    img_predict_id = 0

    child_conn.send('ready')

    while True:
        input_path = child_conn.recv()
        # output = s.predict(d['input_path'])
        output = s.predict(input_path)

        # with lock:
        seg_img = output*255.0
        output_prefix = '{}/{:03d}'.format(predict_dir, img_predict_id)
        output_path = output_prefix + '.png' 
        # print('Annotate finish, save to ' + output_path)
        

        cv2.imwrite( output_path , output )
        cv2.imwrite( output_prefix + '_show.jpg' , seg_img )

        print('copy '. input_path ,' to ', output_prefix + '_ori.png')
        copyfile(input_path, output_prefix + '_ori.png')

        child_conn.send( output_path)

        img_predict_id = (img_predict_id + 1) if img_predict_id < 1000 else 0
	        # cv2.imshow('label', seg_img)

# img_dir = 'semantic_img_' + str( time.time() ) 
# img_dir  = os.path.abspath(img_dir)
# os.makedirs(img_dir)
# print("Make dir: " + img_dir)

parent_conn, child_conn = Pipe()
manager = Manager()

p = Process(target=process_func, args=(child_conn, manager.dict()   )   )
p.daemon = True
p.start()

if parent_conn.recv()=='ready':
    
    #-----test.png----#
    img_path = 'test.png' 
    img_path = os.path.abspath(img_path)
    parent_conn.send(img_path)
    get_path = parent_conn.recv()

    annot_img = cv2.imread(get_path, 1)
    cv2.imshow('annot', annot_img*255.0 )
    cv2.imwrite(  'annot_test.png',  annot_img*255.0 )

    cv2.waitKey(1000)

    #-----test_2.jpg----#
    img_path = 'test_2.jpg' 
    img_path = os.path.abspath(img_path)
    parent_conn.send(img_path)
    get_path = parent_conn.recv()

    annot_img = cv2.imread(get_path, 1)
    cv2.imshow('annot', annot_img*255.0 )
    cv2.imwrite(  'annot_test2.png',  annot_img*255.0 )
    cv2.waitKey(1000)