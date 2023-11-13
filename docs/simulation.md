# Simulation record

- 20231109

- TODO: testset CIFAR10으로 변경가능하게
- Bandwidth Ratio를 Semantic PAPR과 동일하게 1/12

```sh
python train.py --training --trainset CIFAR10 --testset kodak --distortion-metric MSE --model WITT_W/O --channel-type awgn --C 96 --multiple-snr 10
```

- 20231110

```sh
python train.py --training --trainset DIV2K --testset kodak --distortion-metric MSE --model WITT_W/O --channel-type awgn --C 96 --multiple-snr 10
```

- 20231113
- fft가 누락되어 있었음... 추가 완료

```sh
# epoch 50, learning rate 1e-3
DATE=20231113 && nohup python train.py --training --trainset CIFAR10 --testset kodak --distortion-metric MSE --model WITT_W/O --channel-type awgn --C 8 --multiple-snr 10 >> logs/${DATE}.out 2>&1 &
# learning rate 1e-4
DATE=20231113 && nohup python train.py --training --trainset CIFAR10 --testset kodak --distortion-metric MSE --model WITT_W/O --channel-type awgn --C 8 --multiple-snr 10 >> logs/${DATE}-lr-1e-4.out 2>&1 &
disown
# batch size 변경해보기
```
