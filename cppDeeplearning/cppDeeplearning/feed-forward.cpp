#include <iostream>

using namespace std;

#define MAX2(a, b) (a) > (b) ? (a) : (b)

class Neuron
{
public:
	double w_; // weight of one input
	double b_; // bias

	double getAct(const double& x)
	{
		// for linear or identity activation fucntions
		return x;

		// for ReLU activation functions
		//return MAX2(0.0, x)
	}

	double feedForward(const double& input)
	{
		// output_y = f(\sigma)
		// \sigma = w_ * input_x + b
		// for multiple inputs,
		// \sigma = w0_* x0_ + w1_ * x1_ + ... + b

		const double sigma = w_ * input + b_;

		return getAct(sigma);
	}
};

int main() {
	Neuron my_neuron;

	my_neuron.w_ = 2.0;
	my_neuron.b_ = 1.0;

	cout << "Input = 1.0 " << my_neuron.feedForward(1.0) << endl;
	cout << "Input = 2.0 " << my_neuron.feedForward(2.0) << endl;
	cout << "Input = 3.0 " << my_neuron.feedForward(3.0) << endl;

	return 0;
}