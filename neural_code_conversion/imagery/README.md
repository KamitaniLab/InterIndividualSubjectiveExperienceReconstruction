This section provides scripts for the neural code conversion during mental imagery.

The imagery and attention experiments use the same subject pairs. If you have already trained the converters for the attention experiment, you can reuse them for imagery testing. Alternatively, you can use the pretrained converters provided with this project.

To train the neural code converters using content loss for subject pairs, execute:
```sh
python NCC_train.py --cuda
```
- **Note**: Use the `--cuda` flag when running on a GPU server. Omit `--cuda` if training on a CPU server.

To test the neural code converters using content loss for subject pairs, execute:
```sh
python NCC_test.py --cuda
```
