#!/bin/bash
# Google Drive datasets: bash download.sh hqevfi | erf   (HS-ERGB and BS-ERGB need the request forms, see README)
set -e
pip install -q gdown
get() {
  mkdir -p "raw/$1" && cd "raw/$1" && gdown "$2"
  for f in *.zip; do [ -e "$f" ] && unzip -q "$f" && rm "$f"; done
  for f in *.tar *.tar.gz *.tgz; do [ -e "$f" ] && tar xf "$f" && rm "$f"; done
  cd - > /dev/null
}
case "$1" in
  hqevfi) get hqevfi 104ZMJ-M_frImOOCGfLk_HDb2FV1trveT ;;
  erf) get erf/train 1Bsf9qreziPcVEuf0_v3kjdPUh27zsFXK; get erf/test 1Dk7jVQD29HqRVV11e8vxg5bDOh6KxrzL ;;
  *) echo "usage: bash download.sh hqevfi|erf"; exit 1 ;;
esac
