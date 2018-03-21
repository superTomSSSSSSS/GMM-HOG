#include<iostream>
#include<stdio.h>
#include<stack>
#include <math.h>
#include<opencv2\opencv.hpp>
#define X_START 2000
#define X_END 0
#define Y_START 2000
#define Y_END 0
#define MAX_DETECT_BLOCK_NUM 3000           //�������ĳ�ʼ���ο������Ŀ
#define MAXDETECTIONREGIONS 15              //�����������վ��ο������Ŀ
#define BLACKLINE_WIDTH 3        //ͼƬ��������Ӻ��ߵĿ��

class  Searchalgorithm {
	public:
		Searchalgorithm();                        //��ʼ��Searchalgorithm
		virtual ~Searchalgorithm();               //����Searchalogrothm
		void searchstack(cv::Mat src, int *save);     //ͼƬ���붨λ�׿���������
		void rectangle_replan(int *size, int *sizenew);     //������������ɸѡɾ������������ľ���
		//void rectangle_replan(int *size, int *sizenew);
		bool get_widthandheight(long width,long height);         //��ȡͼƬ�����Լ�������ʶ�ַ�����
		int vaild_num;                            //����ɸѡ�õ��İ׿�����
	private:
		int xStart, xEnd, yStart, yEnd;          //�׿�����ı߽�ֵ
		int xmin[MAX_DETECT_BLOCK_NUM], xmax[MAX_DETECT_BLOCK_NUM], ymin[MAX_DETECT_BLOCK_NUM], ymax[MAX_DETECT_BLOCK_NUM];  //�׿�����ı߽�����
		int block_num;                           //�׿���������
		long nHeight;                             //ͼ���
		long nWidth;                              //ͼ���
		int **pbFlag;                            //ͼ���ʶ
		std::stack<int>mystack;                  //������ջʵ�ֵݹ�����
		void rectangleupdate(int *size, int *sizenew);      //�����������
		void fourneighbourhood(int x, int y);     //�����������㷨
		int contain(int xs, int xe, int ys, int ye, int xs1, int xe1, int ys1, int ye1);  //���ΰ������ε����
		int TemplateMatch(unsigned char *src,int *posi,unsigned char *itemplate,int twidth,int theight,int *x,int *y);
};