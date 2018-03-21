#include"my_background_segm.h"
#include<opencv2\opencv.hpp>
using namespace cv;
static const int defaultNMixtures = 5;//默认混合模型个数 等价于之后K值
static const int defaultHistory = 200;//默认历史帧数
static const double defaultBackgroundRatio = 0.7;//默认背景门限
static const double defaultVarThreshold = 2.5*2.5;//默认方差门限
static const double defaultNoiseSigma = 30 * 0.5;//默认噪声方差
static const double defaultInitialWeight = 0.05;//默认初始权值

namespace cv
{
	//腐蚀膨胀
	void refineSegment(Mat& mask, Mat& dst)
	{
		int niters = 6;
		vector<vector<Point>>contours;
		vector<Vec4i>hierarchy;
		dilate(mask, dst, Mat(), Point(-1, -1), niters);
		erode(mask, dst, Mat(), Point(-1, -1), niters);
		dilate(mask, dst, Mat(), Point(-1, -1), niters);
	}
}

//不带参数的构造函数，使用默认值  
my_BackgroundSubtractorMOG::my_BackgroundSubtractorMOG()
{
	frameSize = Size(0, 0);
	frameType = 0;

	nframes = 0;
	nmixtures = defaultNMixtures;
	history = defaultHistory;
	varThreshold = defaultVarThreshold;
	backgroundRatio = defaultBackgroundRatio;
	noiseSigma = defaultNoiseSigma;
}
//带参数的构造函数，使用参数传进来的值    
my_BackgroundSubtractorMOG::my_BackgroundSubtractorMOG(int _history, int _nmixtures,
	double _backgroundRatio,
	double _noiseSigma)
{
	frameSize = Size(0, 0);
	frameType = 0;

	nframes = 0;
	nmixtures = min(_nmixtures > 0 ? _nmixtures : defaultNMixtures, 8);//不能超过8个，否则就用默认的
	history = _history > 0 ? _history : defaultHistory;//不能小于0，否则就用默认的
	varThreshold = defaultVarThreshold;//门限使用默认的
	backgroundRatio = min(_backgroundRatio > 0 ? _backgroundRatio : 0.95, 1.);//背景门限必须大于0，小于1，否则使用0.95
	noiseSigma = _noiseSigma <= 0 ? defaultNoiseSigma : _noiseSigma;//噪声门限大于0
}

my_BackgroundSubtractorMOG::~my_BackgroundSubtractorMOG()
{
}


void my_BackgroundSubtractorMOG::initialize(Size _frameSize, int _frameType)
{
	frameSize = _frameSize;
	frameType = _frameType;
	nframes = 0;

	int nchannels = CV_MAT_CN(frameType);
	CV_Assert(CV_MAT_DEPTH(frameType) == CV_8U);

	// for each gaussian mixture of each pixel bg model we store ...
	// the mixture sort key (w/sum_of_variances), the mixture weight (w),
	// the mean (nchannels values) and
	// the diagonal covariance matrix (another nchannels values)
	bgmodel.create(1, frameSize.height*frameSize.width*nmixtures*(2 + 2 * nchannels), CV_32F);//初始化一个1行*XX列的矩阵
	//XX是这样计算的：图像的行*列*混合模型的个数*（1（优先级） + 1（权值） + 2（均值 + 方差） * 通道数）
	bgmodel = Scalar::all(0);//清零
}

//设为模版，就是考虑到了彩色图像与灰度图像两种情况    
template<typename VT> struct MixData
{
	float sortKey;    //优先级
	float weight;
	VT mean;
	VT var;
};


static void process8uC1(const Mat& image, Mat& fgmask, double learningRate,
	Mat& bgmodel, int nmixtures, double backgroundRatio,
	double varThreshold, double noiseSigma)
{
	int x, y, k, k1, rows = image.rows, cols = image.cols;
	float alpha = (float)learningRate, T = (float)backgroundRatio, vT = (float)varThreshold;//学习速率、背景门限、方差门限
	int K = nmixtures;//混合模型个数
	MixData<float>* mptr = (MixData<float>*)bgmodel.data;

	const float w0 = (float)defaultInitialWeight;//初始权值
	const float sk0 = (float)(w0 / (defaultNoiseSigma * 2));//初始优先级
	const float var0 = (float)(defaultNoiseSigma*defaultNoiseSigma * 4);//初始方差
	const float minVar = (float)(noiseSigma*noiseSigma);//最小方差

	for (y = 0; y < rows; y++)
	{
		const uchar* src = image.ptr<uchar>(y);
		uchar* dst = fgmask.ptr<uchar>(y);

		if (alpha > 0)//如果学习速率为0，则退化为背景相减
		{
			for (x = 0; x < cols; x++, mptr += K)
			{
				float wsum = 0;
				float pix = src[x];//每个像素
				int kHit = -1, kForeground = -1;//是否属于模型，是否属于前景

				for (k = 0; k < K; k++)//每个高斯模型
				{
					float w = mptr[k].weight;//当前模型的权值
					wsum += w;//权值累加
					if (w < FLT_EPSILON)
						break;
					float mu = mptr[k].mean;//当前模型的均值
					float var = mptr[k].var;//当前模型的方差
					float diff = pix - mu;//当前像素与模型均值之差
					float d2 = diff*diff;//平方


					//混合模型的更新
					if (d2 < vT*var)//是否小于方门限，即是否属于该模型   vT
					{
						wsum -= w;//如果匹配，则把它减去，因为之后会更新它
						float dw = alpha*(1.f - w);   //更新新的权值
						mptr[k].weight = w + dw;//增加权值
						//注意源文章中涉及概率的部分多进行了简化，将概率变为1
						mptr[k].mean = mu + alpha*diff;//修正均值       
						var = max(var + alpha*(d2 - var), minVar);//开始时方差清零0，所以这里使用噪声方差作为默认方差，否则使用上一次方差
						mptr[k].var = var;//修正方差
						mptr[k].sortKey = mptr[k].weight / sqrt(var);//重新计算优先级，貌似这里不对，应该使用更新后的mptr[k].weight而不是w

						for (k1 = k - 1; k1 >= 0; k1--)//从匹配的第k个模型开始向前比较，如果更新后的单高斯模型优先级超过他前面的那个，则交换顺序
						{
							if (mptr[k1].sortKey >= mptr[k1 + 1].sortKey)//如果优先级没有发生改变，则停止比较
								break;
							std::swap(mptr[k1], mptr[k1 + 1]);//交换它们的顺序，始终保证优先级最大的在前面
						}

						kHit = k1 + 1;//记录属于哪个模型
						break;
					}
				}

				if (kHit < 0) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
					//当前像素不属于任何一个模型
				{
					//初始化一个新模型
					kHit = k = min(k, K - 1);//有两种情况，当最开始的初始化时，k并不是等于K-1的
					wsum += w0 - mptr[k].weight;//从权值总和中减去原来的那个模型，并加上默认权值
					mptr[k].weight = w0;//初始化权值
					mptr[k].mean = pix;//初始化均值
					mptr[k].var = var0; //初始化方差
					mptr[k].sortKey = sk0;//初始化权值
				}
				else
				for (; k < K; k++)
					wsum += mptr[k].weight;//求出剩下的总权值

				float wscale = 1.f / wsum;//归一化
				wsum = 0;
				for (k = 0; k < K; k++)
				{
					wsum += mptr[k].weight *= wscale;
					mptr[k].sortKey *= wscale;//计算归一化权值

					//通过这个来判断是否为前景还是背景
					if (wsum > T && kForeground < 0)           //T 等价于backgroundRatio 等于0.7
						kForeground = k + 1;//第几个模型之后就判为前景了
				}

				dst[x] = (uchar)(-(kHit >= kForeground));//判决：(ucahr)(-true) = 255;(uchar)(-(false)) = 0;
			}
		}
		else//如果学习速率小于等于0，则没有背景更新过程，其他过程类似
		{
			for (x = 0; x < cols; x++, mptr += K)
			{
				float pix = src[x];
				int kHit = -1, kForeground = -1;

				for (k = 0; k < K; k++)
				{
					if (mptr[k].weight < FLT_EPSILON)
						break;
					float mu = mptr[k].mean;
					float var = mptr[k].var;
					float diff = pix - mu;
					float d2 = diff*diff;
					if (d2 < vT*var)
					{
						kHit = k;
						break;
					}
				}

				if (kHit >= 0)
				{
					float wsum = 0;
					for (k = 0; k < K; k++)
					{
						wsum += mptr[k].weight;
						if (wsum > T)
						{
							kForeground = k + 1;
							break;
						}
					}
				}

				dst[x] = (uchar)(kHit < 0 || kHit >= kForeground ? 255 : 0);
			}
		}
	}
}


static void process8uC3(const Mat& image, Mat& fgmask, double learningRate,
	Mat& bgmodel, int nmixtures, double backgroundRatio,
	double varThreshold, double noiseSigma)
{
	int x, y, k, k1, rows = image.rows, cols = image.cols;
	float alpha = (float)learningRate, T = (float)backgroundRatio, vT = (float)varThreshold;
	int K = nmixtures;

	const float w0 = (float)defaultInitialWeight;
	const float sk0 = (float)(w0 / (defaultNoiseSigma * 2 * sqrt(3.)));
	const float var0 = (float)(defaultNoiseSigma*defaultNoiseSigma * 4);
	const float minVar = (float)(noiseSigma*noiseSigma);
	MixData<float>* mptr = (MixData<float>*)bgmodel.data;

	for (y = 0; y < rows; y++)
	{
		const uchar* src = image.ptr<uchar>(y);
		uchar* dst = fgmask.ptr<uchar>(y);

		if (alpha > 0)
		{
			for (x = 0; x < cols; x++, mptr += K)
			{
				float wsum = 0;
				float pix = src[x];
				int kHit = -1, kForeground = -1;

				for (k = 0; k < K; k++)
				{
					float w = mptr[k].weight;
					wsum += w;
					if (w < FLT_EPSILON)
						break;
					float mu = mptr[k].mean;
					float var = mptr[k].var;
					float diff = pix - mu;
					float d2 = diff*diff;
					if (d2 < vT*var)
					{
						wsum -= w;
						float dw = alpha*(1.f - w);
						mptr[k].weight = w + dw;
						mptr[k].mean = mu + alpha*diff;
						var = max(var + alpha*(d2 - var), minVar);
						mptr[k].var = var;
						mptr[k].sortKey = w / sqrt(var);

						for (k1 = k - 1; k1 >= 0; k1--)
						{
							if (mptr[k1].sortKey >= mptr[k1 + 1].sortKey)
								break;
							std::swap(mptr[k1], mptr[k1 + 1]);
						}

						kHit = k1 + 1;
						break;
					}
				}

				if (kHit < 0) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
				{
					kHit = k = min(k, K - 1);
					wsum += w0 - mptr[k].weight;
					mptr[k].weight = w0;
					mptr[k].mean = pix;
					mptr[k].var = var0;
					mptr[k].sortKey = sk0;
				}
				else
				for (; k < K; k++)
					wsum += mptr[k].weight;

				float wscale = 1.f / wsum;
				wsum = 0;
				for (k = 0; k < K; k++)
				{
					wsum += mptr[k].weight *= wscale;
					mptr[k].sortKey *= wscale;
					if (wsum > T && kForeground < 0)
						kForeground = k + 1;
				}

				dst[x] = (uchar)(-(kHit >= kForeground));
			}
		}
		else
		{
			for (x = 0; x < cols; x++, mptr += K)
			{
				float pix = src[x];
				int kHit = -1, kForeground = -1;

				for (k = 0; k < K; k++)
				{
					if (mptr[k].weight < FLT_EPSILON)
						break;
					float mu = mptr[k].mean;
					float var = mptr[k].var;
					float diff = pix - mu;
					float d2 = diff*diff;
					if (d2 < vT*var)
					{
						kHit = k;
						break;
					}
				}

				if (kHit >= 0)
				{
					float wsum = 0;
					for (k = 0; k < K; k++)
					{
						wsum += mptr[k].weight;
						if (wsum > T)
						{
							kForeground = k + 1;
							break;
						}
					}
				}

				dst[x] = (uchar)(kHit < 0 || kHit >= kForeground ? 255 : 0);
			}
		}
	}
}

void my_BackgroundSubtractorMOG::operator()(InputArray _image, OutputArray _fgmask, double learningRate)
{
	Mat image = _image.getMat();
	bool needToInitialize = nframes == 0 || learningRate >= 1 || image.size() != frameSize || image.type() != frameType;//是否需要初始化

	if (needToInitialize)
		initialize(image.size(), image.type());//初始化 最开始的时候执行一次

	CV_Assert(image.depth() == CV_8U);
	_fgmask.create(image.size(), CV_8U);
	Mat fgmask = _fgmask.getMat();

	++nframes;
	learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1. / min(nframes, history);
	CV_Assert(learningRate >= 0);

	if (image.type() == CV_8UC1)//处理灰度图像
		process8uC1(image, fgmask, learningRate, bgmodel, nmixtures, backgroundRatio, varThreshold, noiseSigma);
	else if (image.type() == CV_8UC3)//处理彩色图像
		process8uC3(image, fgmask, learningRate, bgmodel, nmixtures, backgroundRatio, varThreshold, noiseSigma);
	else
		CV_Error(CV_StsUnsupportedFormat, "Only 1- and 3-channel 8-bit images are supported in my_BackgroundSubtractorMOG");
}
