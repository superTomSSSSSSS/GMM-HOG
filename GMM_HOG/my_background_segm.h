#include "opencv2/core/core.hpp"
#include <list>
#include "opencv2/video/background_segm.hpp"    //找到你自己安装包中该文件的位置

namespace cv
{
	void refineSegment(Mat& mask, Mat& dst);

	class CV_EXPORTS_W my_BackgroundSubtractorMOG : public BackgroundSubtractor
	{
	public:
		//! the default constructor
		CV_WRAP my_BackgroundSubtractorMOG();
		//! the full constructor that takes the length of the history, the number of gaussian mixtures, the background ratio parameter and the noise strength
		CV_WRAP my_BackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma = 0);
		//! the destructor
		virtual ~my_BackgroundSubtractorMOG();
		//! the update operator
		virtual void operator()(InputArray image, OutputArray fgmask, double learningRate = 0);

		//! re-initiaization method
		virtual void initialize(Size frameSize, int frameType);

		// virtual AlgorithmInfo* info() const;

	protected:
		//public:
		Size frameSize;
		int frameType;
		Mat bgmodel;
		int nframes;
		int history;
		int nmixtures;
		double varThreshold;
		double backgroundRatio;
		double noiseSigma;
	};
}
