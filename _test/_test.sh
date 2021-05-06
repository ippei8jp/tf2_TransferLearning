# サンプル画像ダウンロード
if [ ! -e a.jpg ]; then
  wget https://cdn.amebaowndme.com/madrid-prd/madrid-web/images/sites/483796/1357355de6edbc4c4b54d22faf0b0756_ce052e9b134a9dbb047a8e17c890832a.jpg -O a.jpg
  wget https://cdn.amebaowndme.com/madrid-prd/madrid-web/images/sites/483796/564b6ca69e9022aa1977f335a148a05a_2d642c807aaf8f5b972a0a406903447d.jpg -O b.jpg
fi 
python hand_detect.py a.jpg
# python hand_detect.py b.jpg
