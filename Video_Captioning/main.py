from __future__ import print_function, division
from builtins import range

import json
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

from utils import *
from beamsearch import *
from model import Seq2seq
import argparse
import _pickle as pickle 
import time

def load_data(video_path, feat_path, frame_size, feauture_size, caption_path=None):
    # 'data/train_videoid.pkl', 'data/train_caption.pkl'

    with open (video_path, 'rb') as fp:
        train_videoid = pickle.load(fp)
    # print(len(train_videoid))
    train_features = np.zeros((len(train_videoid), frame_size, feauture_size))

    for i in range(0, len(train_videoid)):
        filepath = feat_path + train_videoid[i] + '.npy'
        train_features[i] = np.load(filepath)

    if caption_path is not None:
        with open (caption_path, 'rb') as fp:
           Y = pickle.load(fp)

        print(train_features.shape, len(Y))
        return train_features, Y
    else:
        print(train_features.shape)
        return train_features, train_videoid

def load_data_dict(video_path, feat_path, frame_size, feauture_size, load_y=True):
    
    with open (video_path, 'rb') as fp:
        train_dict = pickle.load(fp)
    # print(len(train_videoid))

    train_features = {}

    for idx, caps in train_dict.items():
        filepath = feat_path + idx + '.npy'
        train_features[idx] = np.load(filepath)

    print(len(train_features), len(train_dict))
    return train_features, train_dict
    

def test(dataset, args, test_id):
    if args.model == 'seq2seq':
        model = Seq2seq(32, args.lr, dataset.vocabsize, args.embed_dim, args.fs, args.feat_dim, dataset.pretrain_wordemb, False, pred_batch_size=1)
    else:
        print('choose a model to train! ')
        parser.print_help()
        return

    model.build_model()
    print(model.x)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    latest_ckpt = tf.train.latest_checkpoint(args.loadpath)
    saver.restore(sess, latest_ckpt)
    print('restore from', latest_ckpt)
    
    fwirte = open(args.output, 'w')

    for step in range(0, dataset.iters):
        batch_x = dataset.next_batch_test()
        hist = np.zeros((1, dataset.vocabsize), dtype=np.float32)
        hist = list(hist)
        indices = np.ones((1, 1), dtype=np.float32)
        indices = list(indices)

        _, probs = sess.run([model.generated_words, model.probs], 
            feed_dict={ model.x: batch_x, model.hist: hist, model.indices:indices})
        

        words = beamsearch(probs, dataset.wtoi)

        sentence = ''
        for idx in words:
            if idx != 0 and dataset.itow[idx] == '<eos>':
                break;
            if idx != 0:
                sentence += dataset.itow[idx] + ' '

        # test for special mission
        if test_id[step] == 'klteYv1Uv9A_27_33.avi' or test_id[step] == '5YJaS2Eswg0_22_26.avi' or test_id[step] == 'UbmZAe5u5FI_132_141.avi' \
            or test_id[step] == 'JntMAcTlOF0_50_70.avi' or test_id[step] == 'tJHUH9tpqPg_113_118.avi':
            print(test_id[step], sentence)

        fwirte.write('%s,%s\n' % (test_id[step], sentence))

    fwirte.close()
    print('save test result file as', args.output)

def train(dataset, args, retrain):
    if args.model == 'seq2seq':
        model = Seq2seq(args.bs, args.lr, dataset.vocabsize, args.embed_dim, args.fs, args.feat_dim, dataset.pretrain_wordemb)
    else:
        print('choose a model to train! ')
        parser.print_help()
        return
    
    model.build_model()

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    if retrain == True:
        latest_ckpt = tf.train.latest_checkpoint(args.loadpath)
        saver.restore(sess, latest_ckpt)
        print('restore from', latest_ckpt)

    print ('Start training, method=%s, lr=%f, epoch=%d, comment=%s'% (model.name, args.lr, args.ep, args.comment))

    path_args = args.savepath.split('/')[0] + '/output.log'
    fwirte = open(path_args, 'w')

    preloss = 1000.
    earlystop = 0
    cutting_len = 8

    pre_loss = 10.0
    roll_sch = 1.0
    for ep in range(1, args.ep+1):
        correct = []
        train_total_loss = 0

        start_time = time.time()

        if pre_loss < 0.65 and cutting_len < 45: # prevent overfitting
            cutting_len += 10
            dataset.random_sample(4, int(cutting_len))
            print('re sample, size:', dataset.size, ', iters:', dataset.iters, 'cutting_size:', cutting_len)

        else:
            dataset.random_sample(4, int(cutting_len))
        
        for step in range(0, dataset.iters):          # total run 21120 samples
            batch_x, batch_y = dataset.next_batch() #dataset.next_batch() 
            hist = np.zeros((batch_x.shape[0], dataset.vocabsize), dtype=np.float32)
            hist = list(hist)
            indices = np.ones((batch_x.shape[0], 1), dtype=np.float32)
            indices = list(indices)

            batch_y_mask = np.zeros( (batch_y.shape[0], batch_y.shape[1]) )
            nonzeros = list( map(lambda x: (x != 0).sum() , batch_y ) )
            for ind, row in enumerate(batch_y_mask):
                row[:nonzeros[ind]] = 1
            
            if step == 0 and ep == 1:
                print(batch_x.shape, batch_y.shape)
            roll = np.random.rand()
            _, loss = sess.run([model.train_op, model.loss_op], 
                feed_dict={ model.x: batch_x, model.caption: batch_y, model.caption_mask: batch_y_mask, 
                            model.prob_sch:roll_sch, model.roll:roll, model.hist:hist, model.indices:indices})
            # print(logit_words)
            train_total_loss += loss

            if step % 10 == 0 or step == 1:
                # pred, current_embed = sess.run([model.pred, model.current_embed], 
                #     feed_dict={ model.x: batch_x, model.caption: batch_y}),
                # print(len(pred))
                # print(pred[0])
                # print('================================\n')
                # print(len(current_embed))
                # print(current_embed[0])
                print("Epoch: %2d, Step: %7d/%7d, Train_loss: %.4f, roll: %2.3f, roll_sch: %2.3f         " % 
                    (ep, step, dataset.iters, loss, roll, roll_sch), end='\r')

        train_total_loss /= dataset.iters
        pre_loss = train_total_loss

        print("Epoch: %2d, Step: %7d/%7d, Train_loss: %2.4f        " % 
                    (ep, step, dataset.iters, train_total_loss), end='\r')

        test_total_loss = 0
        # total_iters = 75
        # totalsample = dataset.random_sample(2400)
        # total_iters = np.ceil(totalsample/args.bs).astype(np.int32)
        dataset.random_sample(1)
        for step in range(0, dataset.iters): 
            batch_x, batch_y = dataset.next_batch() #dataset.next_batch_val()
            hist = np.zeros((batch_x.shape[0], dataset.vocabsize), dtype=np.float32)
            hist = list(hist)
            indices = np.ones((batch_x.shape[0], 1), dtype=np.float32)
            indices = list(indices)
            # batch_y = np.column_stack((batch_y, np.zeros( [len(batch_y), 1] ))).astype(int)
            batch_y_mask = np.zeros( (batch_y.shape[0], batch_y.shape[1]) )
            nonzeros = list( map(lambda x: (x != 0).sum() , batch_y ) )
            for ind, row in enumerate(batch_y_mask):
                row[:nonzeros[ind]] = 1
            roll = 0.0
            loss = sess.run(model.loss_op, 
                feed_dict={ model.x: batch_x, model.caption: batch_y, model.caption_mask: batch_y_mask, 
                            model.prob_sch:roll_sch, model.roll:roll, model.hist:hist, model.indices:indices})

            test_total_loss += loss

        test_total_loss /= dataset.iters
        end_time = time.time()
        print("Epoch: %2d, take_time: %4.1fs, Train_loss: %2.4f, Test_loss: %2.4f          " % 
                    (ep, (end_time-start_time), train_total_loss, test_total_loss))
        fwirte.write("Epoch: %2d, take_time: %4.1fs, Train_loss: %2.4f, Test_loss: %2.4f\n" % 
                    (ep, (end_time-start_time), train_total_loss, test_total_loss))

        saver.save(sess, args.savepath, global_step=ep)
        if ep > 50:
            roll_sch *= 0.99

        if test_total_loss < 0.55:
            print('earlystop at epoch %d' %(ep))
            break


    print('Done')
    print("Model saved in file: %s\n" % args.savepath)
    fwirte.write('Done')
    fwirte.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='set this to train.')
    parser.add_argument('--retrain', action='store_true', help='set this to retrain.')
    parser.add_argument('--test', action='store_true', help='set this to testing.')
    parser.add_argument('--peer', action='store_true', help='set this to testing.')
    parser.add_argument('--model', metavar='', type=str, default="seq2seq", help='choose a model to train/test.')

    parser.add_argument('--lr', metavar='', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=100, help='number of epochs.')
    parser.add_argument('--bs', metavar='', type=int, default=32, help='batch size.')
    parser.add_argument('--feat_dim', metavar='', type=int, default=4096, help='features of video (input).')
    parser.add_argument('--fs', metavar='', type=int, default=80, help='frame size of video (input).')
    parser.add_argument('--embed_dim', metavar='', type=int, default=300, help='dim of word embedding')
    # parser.add_argument('--tstep', metavar='', type=int, default=123, help='sequence len.')
    parser.add_argument('--savepath', metavar='', type=str, default="", help='save/load model path.') # model_rnn/model.ckpt
    parser.add_argument('--loadpath', metavar='', type=str, default="", help='save/load model path.') # model_rnn/
    parser.add_argument('--input', metavar='', type=str, default="../data/MLDS_hw2_data/", help='input data path.')
    parser.add_argument('--comment', metavar='', type=str, default="", help='describe this model')
    parser.add_argument('--output', metavar='', type=str, default="result.txt", help='the path to save output file')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))

    if args.train or args.retrain:
        if args.savepath is None :
            print('give args --savepath yourpath')
        else:
            # first train_caption_fewer, train_id_fewer
            # then train_caption_longer, train_id_longer
            start = time.time()
            feat_path = args.input + 'training_data/feat/'
            train_features, train_dict = load_data_dict('data/train_caption_dict.pkl', feat_path, args.fs, args.feat_dim)
            end = time.time()

            print('pre loading raw data, use %4.1fs' % (end-start))

            dataset = DataLoader(train_features, train_dict, args.bs, args.fs, args.embed_dim)
            train(dataset, args, args.retrain)

            objarg = parser.parse_args()
            path_args = args.savepath.split('/')[0] + '/args.txt'
            with open(path_args, 'w') as output_file:
                output_file.write("%s\n" % objarg)

    if args.test:
        if args.loadpath is None:
            print('give args --loadpath yourpath')
        else:
            start = time.time()
            if args.peer:
                feat_path = args.input + 'peer_review/feat/'
                pkl_path = 'data/peer_id.pkl'
            else:
                feat_path = args.input + 'testing_data/feat/'
                pkl_path = 'data/test_id.pkl'

            X, test_id = load_data(pkl_path, feat_path=feat_path, frame_size=args.fs, feauture_size=args.feat_dim)
            end = time.time()
            print('pre loading raw data, use %4.1fs' % (end-start))
            
            dataset = DataLoader(X, batch_size=args.bs, fsize=args.fs, emb_dim=args.embed_dim, test=True)
            test(dataset, args, test_id)

    # if args.special:
    #     if args.loadpath is None:
    #         print('give args --loadpath yourpath')
    #     else:
    #         start = time.time()
    #         feat_path = args.input + 'testing_data/feat/'
    #         X, test_id = load_data('data/special_id.pkl', feat_path=feat_path, frame_size=args.fs, feauture_size=args.feat_dim)
    #         end = time.time()
    #         print('pre loading raw data, use %4.1fs' % (end-start))
            
    #         dataset = DataLoader(X, batch_size=args.bs, fsize=args.fs, emb_dim=args.embed_dim, test=True)
    #         test(dataset, args, test_id)
