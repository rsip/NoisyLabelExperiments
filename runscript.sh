for p in $(seq 0.0 0.2 1.0)
do
  python noisylabels.py -d cifar100 -m shallow_cnn -e most_confusing -p $p
  sleep 30s
done
exit 0
