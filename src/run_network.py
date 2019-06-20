from detector import YoloImgRun, YoloVideoRun, YoloImageStream
from argLoader import ArgLoader


if __name__ == '__main__':
    argLoader = ArgLoader()
    args = argLoader.args  # parse the command line arguments

    run_style = args.run_style
    if run_style == 1:
        detector = YoloVideoRun(args)
    else:
        detector = YoloImageStream(args)
        
    detector.run()