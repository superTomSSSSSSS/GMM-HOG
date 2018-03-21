#include"my_background_segm.h"
#include"searchalgorithm.h"
#include<iostream>
#include<opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

int main()
{
	Mat foreground, frame;
	HOGDescriptor hog;       //HOG���������
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());   //����SVM������ΪĬ�ϲ���
	BackgroundSubtractorMOG mog = BackgroundSubtractorMOG(200, 5, 0.7, 15);      //�ı�0.7����ʹ�ñ������Ŀն�������ʧ
	VideoCapture capture("2_1.mp4");
	if (!capture.isOpened())
	{
		cout << "��ȡ��Ƶʧ��" << endl;
		return -1;
	}
	int mark = 0;
	int nWidth, nHeight;
	while (1)
	{
		Mat src;
		vector<Rect> found, found_filtered;
		int save[MAX_DETECT_BLOCK_NUM] = { 0 };           //������Ƶ�Ĵ�С ���ֵҲ��Ҫ�Ķ�
		int savenew[MAX_DETECT_BLOCK_NUM] = { 0 };
		Searchalgorithm Search;

		capture >> frame;

		if (0 == mark)
		{
			nWidth = frame.cols;
			nHeight = frame.rows;
			mark = 1;
		}

		Search.get_widthandheight(nWidth, nHeight);

		cvtColor(frame, frame, CV_BGR2GRAY);
			
		//����Ϊ������ͼ�����ͼ��ѧϰ����
		mog(frame, foreground, 0.01);       //�ı�ѧϰ���� ���Է�ֹ�������ֿն�  

		cv::refineSegment(foreground, foreground);
		Search.searchstack(foreground, save);
		Search.rectangle_replan(save, savenew);

		//�߽�ļ��
		for (int count = 0; count < Search.vaild_num; count++){         
			int EndX = save[count * 4 + 1] + 30;
			int EndY = save[count * 4 + 3] + 60;
			int initX = save[count * 4 + 0] - 20;
			int initY = save[count * 4 + 2] - 50;
			if (EndX > nWidth)
			{
				EndX = nWidth - 1;
			}
			if (EndY > nHeight)
			{
				EndY = nHeight - 1;
			}
			if (initX <= 0)
			{
				initX = 0;
			}
			if (initY <= 0)
			{
				initY = 0;
			}
			int Width = EndX - initX;
			int Height = EndY - initY;
			rectangle(frame, Point(initX, initY),Point(EndX,EndY), Scalar(100, 255, 90), 3);
			src = frame(Rect(initX, initY, Width, Height));

			hog.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
			for (int i = 0; i < found.size(); i++)
			{
				Rect r = found[i];
				int j = 0;
				for (; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
				if (j == found.size())
					found_filtered.push_back(r);
			}

			for (int i = 0; i < found_filtered.size(); i++)
			{
				Rect r = found_filtered[i];
				r.x += cvRound(r.width*0.1)+initX;
				r.width = cvRound(r.width*0.8);
				r.y += cvRound(r.height*0.07)+initY;
				r.height = cvRound(r.height*0.8);
				rectangle(frame, r.tl(), r.br(), Scalar(0, 255, 0), 3);
			}
		}

		imshow("finallpic", frame);
		waitKey(30);
	}
	return 0;
}