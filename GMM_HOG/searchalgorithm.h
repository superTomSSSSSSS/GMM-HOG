#include<iostream>
#include<stdio.h>
#include<stack>
#include <math.h>
#include<opencv2\opencv.hpp>
#define X_START 2000
#define X_END 0
#define Y_START 2000
#define Y_END 0
#define MAX_DETECT_BLOCK_NUM 3000           //检测出来的初始矩形框最大数目
#define MAXDETECTIONREGIONS 15              //检测出来的最终矩形框最大数目
#define BLACKLINE_WIDTH 3        //图片左右两侧加黑线的宽度

class  Searchalgorithm {
	public:
		Searchalgorithm();                        //初始化Searchalgorithm
		virtual ~Searchalgorithm();               //析构Searchalogrothm
		void searchstack(cv::Mat src, int *save);     //图片输入定位白块区域坐标
		void rectangle_replan(int *size, int *sizenew);     //矩形坐标重新筛选删除包含在里面的矩形
		//void rectangle_replan(int *size, int *sizenew);
		bool get_widthandheight(long width,long height);         //获取图片长宽以及创建标识字符数组
		int vaild_num;                            //最终筛选得到的白块数组
	private:
		int xStart, xEnd, yStart, yEnd;          //白块区域的边界值
		int xmin[MAX_DETECT_BLOCK_NUM], xmax[MAX_DETECT_BLOCK_NUM], ymin[MAX_DETECT_BLOCK_NUM], ymax[MAX_DETECT_BLOCK_NUM];  //白块区域的边界数组
		int block_num;                           //白块区域数量
		long nHeight;                             //图像高
		long nWidth;                              //图像宽
		int **pbFlag;                            //图像标识
		std::stack<int>mystack;                  //创建堆栈实现递归序列
		void rectangleupdate(int *size, int *sizenew);      //矩形坐标更新
		void fourneighbourhood(int x, int y);     //四邻域搜索算法
		int contain(int xs, int xe, int ys, int ye, int xs1, int xe1, int ys1, int ye1);  //矩形包含矩形的情况
		int TemplateMatch(unsigned char *src,int *posi,unsigned char *itemplate,int twidth,int theight,int *x,int *y);
};