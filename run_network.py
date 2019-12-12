#!/usr/bin/env python3
from pytorchYolo.detector import YoloImgRun, YoloVideoRun, YoloImageStream
from pytorchYolo.argLoader import ArgLoader


if __name__ == '__main__':
    argLoader = ArgLoader()
    args = argLoader.args  # parse the command line arguments
    #print(args)
    run_style = args.run_style
    if run_style == 1:
        detector = YoloVideoRun(args)
    else:
        if args.use_batches == 1:
            detector = YoloImgRun(args)
        else:
            detector = YoloImageStream(args)

    detector.run()
