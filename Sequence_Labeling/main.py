from __future__ import print_function, division
from builtins import range
import json
import tensorflow as tf
import numpy as np 
from model import RNN_LSTM, CNN_RNN
from utils import *
import argparse
import pickle 
import time

def pred_correct(y, n_class):
    return np.argmax(np.reshape(y, [-1, n_class]), 1)

def test(testset, args):
    if args.model == 'rnn':
        model = RNN_LSTM(args.bsize, args.tstep, args.n_classes, args.n_input, args.lr, args.n_hidden, False)
    elif args.model == 'cnn':
        model = CNN_RNN(args.bsize, args.tstep, args.n_classes, args.n_input, args.lr, args.n_hidden)
    else:
        print('choose a model to train! ')
        parser.print_help()
        return

    model.build()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver() 
    latest_ckpt = tf.train.latest_checkpoint(args.loadpath)
    saver.restore(sess, latest_ckpt)

    result = []
    t=0
    f_correct = open('comfirm_correct.txt', 'w')
    for step in range(0, testset.iters):
        batch_x = testset.next_batch_test_seq() 

        batch_seq_len = get_sequence_lengths(batch_x) 
        batch_x_seq = np.squeeze(make_sequences_same_length(batch_x, batch_seq_len))

        if step == 0:
            print(batch_x_seq.shape)

        so_max, pred = sess.run([model.so_max, model.pred], 
            feed_dict={ model.x_train: batch_x_seq, 
                        model.seq_len: batch_seq_len})
        count = 0
        for i in range(0, batch_x_seq.shape[0]): # batch
            seq_result = []
            for j in range(0, batch_x_seq.shape[1]): # seq
                if max(so_max[count]) > 0.85:
                    seq_result.append(pred[count])
                else:
                    seq_result.append('eee')
                    t+=1
                count += 1
            for item in seq_result:
                f_correct.write("%s, " % item)
            f_correct.write('\n')
            result.append(seq_result) 

    print("Testing Done ", len(result), t)

    f_correct.close()

    import pickle
    with open('predict', 'wb') as fp:
        pickle.dump(result, fp)

    # output file: 
    # binary format, every line present a sequence prediction represent by numbers
    # In this kaggle competition, it needs to transfer number to phone
 

def train(trainset, testset, args, retrain):
    if args.model == 'rnn':
        model = RNN_LSTM(args.bsize, args.tstep, args.n_classes, args.n_input, args.lr, args.n_hidden)
    elif args.model == 'cnn':
        model = CNN_RNN(args.bsize, args.tstep, args.n_classes, args.n_input, args.lr, args.n_hidden)
    else:
        print('choose a model to train! ')
        parser.print_help()
        return
    
    model.build()

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    if retrain == True:
        latest_ckpt = tf.train.latest_checkpoint(args.loadpath)
        saver.restore(sess, latest_ckpt)
        print('restore from', latest_ckpt)

    print ('Start training, method=%s, lr=%f, timestep=%d, epoch=%d, comment=%s'% (model.name, args.lr, args.tstep, args.ep, args.comment))

    path_args = args.savepath.split('/')[0] + '/output.log'
    fwirte = open(path_args, 'w')

    preloss = 1000.
    earlystop = 0 

    for ep in range(1, args.ep+1): 

        correct = []
        total_loss = 0

        start_time = time.time()

        for step in range(0, trainset.iters):
            batch_x, batch_y = trainset.next_batch_seq() 
            batch_seq_len = get_sequence_lengths(batch_x) 
            batch_x_seq = np.squeeze(make_sequences_same_length(batch_x, batch_seq_len))
            batch_y_seq = np.squeeze(make_sequences_same_length(batch_y, batch_seq_len))

            if step == 0 and ep == 1:
                print(batch_x_seq.shape, batch_y_seq.shape)

            _, loss, pred = sess.run([model.train_op, model.loss_op, model.pred], 
                feed_dict={ model.x_train: batch_x_seq, model.y_train: batch_y_seq, 
                            model.seq_len: batch_seq_len}) 
            pred_correct_y = pred_correct(batch_y_seq, args.n_classes)
            correct = np.append(correct, np.equal(pred, pred_correct_y))

            total_loss += loss

            if step % 10 == 0 or step == 1:
                print("Epoch: %2d, Step: %7d/%7d, Train_loss: %.4f, Train_acc: %.4f %%          " % 
                    (ep, step, trainset.iters, loss, np.mean(np.equal(pred, pred_correct_y)) * 100), end='\r') 

        total_loss /= trainset.iters

        fwirte.write("Epoch: %2d, Step: %7d/%7d, Train_loss: %2.4f, Train_acc: %2.4f %%\n" % 
                         (ep, step, trainset.iters, total_loss, np.mean(np.stack(correct)) * 100))
        print("Epoch: %2d, Step: %7d/%7d, Train_loss: %2.4f, Train_acc: %2.4f %%         " % 
                    (ep, step, trainset.iters, total_loss, np.mean(np.stack(correct)) * 100))

        correct = []
        total_loss = 0
        for step in range(0, testset.iters):
            batch_x, batch_y = testset.next_batch_seq() 
            batch_seq_len = get_sequence_lengths(batch_x) 
            batch_x_seq = np.squeeze(make_sequences_same_length(batch_x, batch_seq_len)) 
            batch_y_seq = np.squeeze(make_sequences_same_length(batch_y, batch_seq_len)) 

            loss, pred = sess.run([model.loss_op, model.pred], 
                feed_dict={ model.x_train: batch_x_seq, model.y_train: batch_y_seq, 
                            model.seq_len: batch_seq_len})
            total_loss += loss
            pred_correct_y = pred_correct(batch_y_seq, args.n_classes)
            correct = np.append(correct, np.equal(pred, pred_correct_y))

        total_loss /= testset.iters 

        end_time = time.time() 

        print("\t\tTake time: %4.1fs, Testing loss: %2.3f Testing accuracy: %2.3f %%" % 
            ( (end_time-start_time), total_loss, np.mean(np.stack(correct)) * 100))
        fwirte.write("\t\tTake time: %4.1fs, Testing loss: %2.3f Testing accuracy: %2.3f %%\n" % 
            ( (end_time-start_time), total_loss, np.mean(np.stack(correct)) * 100))

        saver.save(sess, args.savepath, global_step=ep)

        """
        Early stop
        """
        if preloss < total_loss:
            if earlystop == 0:
                preloss = total_loss # the lowest place
            earlystop += 1
        else:
            earlystop = 0
            preloss = total_loss

        if earlystop >= 10:
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
    parser.add_argument('--model', metavar='', type=str, default="rnn", help='choose a model to train/eval/test.')
    parser.add_argument('--lr', metavar='', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--n_hidden', metavar='', type=int, default=512, help='number of hidden layer in lstm.')
    parser.add_argument('--ep', metavar='', type=int, default=100, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=128, help='batch size.')
    parser.add_argument('--n_classes', metavar='', type=int, default=39, help='number of classes (output).')
    parser.add_argument('--n_input', metavar='', type=int, default=39, help='number of features (input).')
    parser.add_argument('--tstep', metavar='', type=int, default=123, help='timestep.')
    parser.add_argument('--savepath', metavar='', type=str, default="", help='save/load model path.') # model_rnn/model.ckpt
    parser.add_argument('--loadpath', metavar='', type=str, default="", help='save/load model path.') # model_rnn/
    parser.add_argument('--input', metavar='', type=str, default="data/", help='input data path.')
    parser.add_argument('--comment', metavar='', type=str, default="", help='describe this model')
    parser.add_argument('--output', metavar='', type=str, default="", help='the path to save output file')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
     
    if args.train or args.retrain:
        if args.savepath is None :
            print('give args --savepath yourpath')
        else:
            # seq_x : mfcc feature of an audio seq
            # seq_y : mfcc answer of an audio seq
            path_x = args.input + 'mfcc/seq_x.dat'
            path_y = args.input + 'mfcc/seq_y.dat'
            X = np.load(path_x, encoding='bytes') 
            Y = np.load(path_y, encoding='bytes') 
        
            Xtrain = X #[:-360]
            Ytrain = Y #[:-360]
            Xtest  = X[-360:]
            Ytest  = Y[-360:]
            trainset = DataLoader(Xtrain, Ytrain, args.bsize, args.tstep, args.n_classes, build=False)
            valset = DataLoader(Xtest, Ytest, args.bsize, args.tstep, args.n_classes, build=False)
            train(trainset, valset, args, args.retrain)

            objarg = parser.parse_args()
            path_args = args.savepath.split('/')[0] + '/args.txt'
            with open(path_args, 'w') as output_file:
                output_file.write("%s\n" % objarg) 
    if args.test:
        if args.loadpath is None:
            print('give args --loadpath yourpath')
        else:
            path_test = args.input + 'mfcc/seq_test.dat'
            X = np.load(path_test, encoding='bytes')
            testset = DataLoader(X, batch_size=args.bsize, timestep=args.tstep, num_class=args.n_classes, build=False)
            test(testset, args) 
