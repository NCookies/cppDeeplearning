#include <iostream>

using namespace std;

#define MAX2(a, b) (a) > (b) ? (a) : (b)

class Neuron
{
public:
	Neuron()
		: w_(2.0), b_(1.0)
	{}

	Neuron(const double& w_input, const double& b_input)
		: w_(w_input), b_(b_input)
	{}

public:
	double w_; // weight of one input
	double b_; // bias

	double input_, output_;	// saved for back-prop

public:
	double feedForward(const double& _input)
	{
		// output_y = f(\sigma)
		// \sigma = w_ * input_x + b
		// for multiple inputs,
		// \sigma = w0_* x0_ + w1_ * x1_ + ... + b

		input_ = _input;

		const double sigma = w_ * input_ + b_;

		output_ = getAct(sigma);

		return output_;
	}

	void propBackward(const double& target)
	{
		const double alpha = 0.1;  // learning rate
		const double grad = (output_ - target) * getActGrad(output_);

		w_ -= alpha * grad * input_;  // last input_ came from d(wx + b) / dw = x
		b_ -= alpha * grad * 1.0;  // last 1.0 came from d(wx + b) / db = 1
	}

	double getAct(const double& x)
	{
		// for linear or identity activation fucntions
		return x;

		// for ReLU activation functions
		//return MAX2(0.0, x)
	}

	double getActGrad(const double& x)
	{
		// for linear or identity activation fucntions
		return 1.0;

		// for ReLU if (x > 0.0) return 1.0; else 0.0;
		//return MAX2(0.0, x)
	}

	void feedForwardPrint(const double& input)
	{
		printf("%f %f \n", input, feedForward(input));
	}
};

int main() 
{
	// initialize my_neuron
	Neuron my_neuron(2.0, 1.0);

	for (int r = 0; r < 100; r++)
	{
		cout << "Training" << r << endl;
		my_neuron.feedForwardPrint(1.0);
		my_neuron.propBackward(4.0);
		cout << "w_ = " << my_neuron.w_ << "b = " << my_neuron.b_ << endl;
	}

	return 0;
}
