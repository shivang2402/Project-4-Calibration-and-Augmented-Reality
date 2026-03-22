Name: Shivang Patel (shivang2402)

OS: macOS (Apple Silicon, Homebrew OpenCV 4.13.0)
IDE: Terminal + Text Editor

Time Travel Days: Using 1 time travel day (deadline Mar 21, submitting Mar 22)

How to build:
  cd src
  make calibar      # builds the main calibration/AR program
  make features     # builds the feature detection program
  make clean        # removes object files

How to run (calibration/AR):
  cd ..
  ./bin/calibar data          # run on image directory
  ./bin/calibar               # run on live webcam

  Keys:
    s = save current frame for calibration
    c = run calibration (need 5+ saved frames)
    w = write calibration to file
    l = load calibration from file
    p = toggle pose printing
    a = toggle 3D axes
    v = toggle virtual house
    t = toggle tree (extension)
    h = toggle target hiding (extension)
    ] / [ = next/prev image
    z = save screenshot
    q = quit

How to run (feature detection):
  ./bin/features data

  Keys:
    h = Harris corners
    o = ORB features
    + / - = adjust threshold or feature count
    ] / [ = next/prev image
    z = save screenshot
    q = quit

Extensions:
  1. Static image AR (whole system works on pre-captured photos)
  2. Creative virtual scene (house + tree, 5 colors)
  3. Target hiding (green overlay covers checkerboard, press h)

To test extensions:
  ./bin/calibar data
  Press l to load calibration
  Press v for house, t for tree, h to hide target
  Browse with ] and take screenshots with z