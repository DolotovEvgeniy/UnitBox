#include "neural_network.h"
#include <iostream>
using namespace std;
int main(int argc, char** argv) {
    cout << argv[1] << " " << argv[2] << endl;
    NeuralNetwork net(argv[1], argv[2]);
    return  0;
}
