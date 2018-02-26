def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--comment', metavar='', type=str, default="", help='describe this model')
    parser.add_argument('--loadpath', type=str, default="model_pg/savepath", help='load the saved model')

    parser.add_argument('--savepath', type=str, default="", help='load the saved model')


    parser.add_argument('--retrain', action='store_true', help='whether to retrain')
    return parser
