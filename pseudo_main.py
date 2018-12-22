'''
    Pseudo code to define the codebase
'''


    # Task / Dataset args
    parser.add_argument('--datasets', type=list, 
            help='list of datasets in incremental learning order')
    parser.add_argument('--inc', type=str, choices=['icl','idl'],
            help='icl: incremental class learning, idl: incremental dataset learning ')
    parser.add_argument('--task', type=str, choices=['disc','gen'],
            help='if disc: use task labels and focus on supervise learning')
    parser.add_argument('--next', type=str, choices=['perf', 'time', ...],
            help='when to change task ?
                        perf: based on the performance e.g. if acc>95% --> go to next
                        time: after fixed amount of time --> go to next')
                        

    # MODEL args
    parser.add_argument('--rehearsal', type=str, choices=['on','off','pseudo'],
            help='on: rehearse, off: dont, pseudo: rehearse on generative model')
    parser.add_argument('--dgr', type=str, choices=['vae','gan','vampriror?'],
            help='if rehearsal==pseudo, which generative model to use')
    parser.add_argument('--rtf', type=str, choices=['on','off'],
            help='if dgr: combine solver with generator (from https://arxiv.org/pdf/1809.10635.pdf)')

    
    # TRAINING args
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--grad_clip', type=float, default=10.)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--opt', type=str, default=1e-3)
    parser.add_argument('--epochs', type=int, default=None)

    # DATA args

    # OTHER args
    parser.add_argument('--no_cuda', action='store_true')

