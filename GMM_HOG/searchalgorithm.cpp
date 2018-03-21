#include<stdio.h>
#include"searchalgorithm.h"
#include<algorithm>

using namespace std;

//��ʼ�� Searchalgorithm
Searchalgorithm::Searchalgorithm()
{
	xStart = X_START;
	xEnd = X_END;
	yStart = Y_START;
	yEnd = Y_END;
	memset(xmin,0,MAX_DETECT_BLOCK_NUM);            //3000˵��ɨ�������3000����ɫ������򣬳�����ִ�����п����������ֵ��������ɵ�
	memset(xmax,0,MAX_DETECT_BLOCK_NUM);
	memset(ymin,0,MAX_DETECT_BLOCK_NUM);
	memset(ymax,0,MAX_DETECT_BLOCK_NUM);
	block_num = 0;
	nWidth = 0;
	nHeight = 0;
	vaild_num = 0;
	pbFlag = nullptr;
}

//���� Searchalgorithm
Searchalgorithm::~Searchalgorithm(){
	if (pbFlag != nullptr){
		for (int i = 0; i < nHeight; i++)
			delete[] pbFlag[i];
		delete[]pbFlag;
	}
	mystack.empty();
}

//��ȡͼƬ����ʹ�����ʶ�ַ�����
bool Searchalgorithm::get_widthandheight(long width,long height)
{
	nWidth = width;
	nHeight =height;

	if (nWidth == 0 || nHeight == 0)
		return false;

	pbFlag = new int *[nHeight];
	for (int i = 0; i < nHeight; i++){
		pbFlag[i] = new int[nWidth];
	}

	if (pbFlag == nullptr)
		return false;

	return true;
}

//������ݹ��ѵ㷨 ��ջ����ʾ 
void Searchalgorithm::searchstack(cv::Mat src, int *save)
{
	for (int i = 0; i < nHeight; i++)
	{
		uchar *data = src.ptr<uchar>(i);
		for (int j = 0; j < nWidth; j++)
		{
			if (data[j] == 255)
			{
				pbFlag[i][j] = 1;                          //��ɫpbFlag����true
			}
			else
			{
				pbFlag[i][j] = 0;
			}
		}
	}

	for (int a = 0; a < nHeight; a++){
		for (int b = 0; b < nWidth; b++){
			xStart = X_START;               //������ֵҪ��ͼƬ��С�仯 ����   Ҫ��ͼ�����صĸ��Ӵ�
			xEnd = X_END;
			yStart = Y_START;
			yEnd = Y_END;
			if (pbFlag[a][b] == 1)
			{
				fourneighbourhood(b, a);
				//��������������ж���Щ������ ��Χ��û����ͨ�׵�������
				if (xStart == X_START && xEnd == X_END && yStart == Y_START && yEnd == Y_END)
				{
					xStart = 0;            
					xEnd = 0;
					yStart = 0;
					yEnd = 0;
				}
				//����ɫ�������С���X��Y���긳ֵ�������б�������
				xmin[block_num] = xStart;
				xmax[block_num] = xEnd;
				ymin[block_num] = yStart;
				ymax[block_num] = yEnd;

				//ѡ������С��400����x���������С��x����С�������y����������ȥy����С����С��90ʱ,�Զ�ʡȥ��ɾѡ��׼��
				if ((xmax[block_num] - xmin[block_num]) * (ymax[block_num] - ymin[block_num]) < (nWidth * nHeight / 200) || (xmax[block_num] - xmin[block_num]) <= 0 || (ymax[block_num] - ymin[block_num]) <= 87)
				{
					xmin[block_num] = X_START;
					xmax[block_num] = X_END;
					ymin[block_num] = Y_START;
					ymax[block_num] = Y_END;
				}
				if ((xmin[block_num] != X_START) && (xmax[block_num] != X_END) && (ymin[block_num] != Y_START) && (ymax[block_num] != Y_END)){
					save[vaild_num * 4 + 0] = xmin[block_num];
					save[vaild_num * 4 + 1] = xmax[block_num];
					save[vaild_num * 4 + 2] = ymin[block_num];
					save[vaild_num * 4 + 3] = ymax[block_num];
					vaild_num++;
				}
				block_num++;
				if(block_num>=MAX_DETECT_BLOCK_NUM)return;
			}
		}
	}
}

//������������
void Searchalgorithm::fourneighbourhood(int x, int y)
{
	if (pbFlag[y][x] == 1)
		pbFlag[y][x] = 0;
	while (((x > 0) && pbFlag[y][x - 1] == 1) || (x < (nWidth-1) && pbFlag[y][x + 1] == 1) || (y > 0 && pbFlag[y - 1][x] == 1) || (y < (nHeight-1) && pbFlag[y + 1][x] == 1))
	{
		if (((x > 0) && pbFlag[y][x - 1] == 1))
		{
			//��¼����Ĵ�С
			xStart = min(xStart, (x - 1));
			yStart = min(yStart, y);
			mystack.push(x - 1);
			mystack.push(y);
			x = x - 1;
			pbFlag[y][x] = 0;
		}

		//��ջ����ʾ

		//�������ұߵĵ�

		else if ((x < (nWidth-1)) && pbFlag[y][x + 1] == 1)
		{
			//��¼����Ĵ�С
			xEnd = max(xEnd, (x + 1));
			yStart = min(yStart, y);
			mystack.push(x + 1);
			mystack.push(y);
			x = x + 1;
			pbFlag[y][x] = 0;
		}

		//���������ϵĵ�

		else if (y > 0 && pbFlag[y - 1][x] == 1)
		{
			//��¼����Ĵ�С
			yStart = min(yStart, (y - 1));
			mystack.push(x);
			mystack.push(y - 1);
			y = y - 1;
			pbFlag[y][x] = 0;
		}

		// �������±ߵĵ� 

		else if (y < (nHeight-1) && pbFlag[y + 1][x] == 1)
		{
			// ��¼����Ĵ�С 
			yEnd = max(yEnd, (y + 1));
			mystack.push(x);
			mystack.push(y + 1);
			y = y + 1;
			pbFlag[y][x] = 0;
		}
		else{
			pbFlag[y][x] = 0;
		}
	}
	while (!mystack.empty())
	{
		int y = mystack.top();
		mystack.pop();
		int x = mystack.top();
		mystack.pop();
		while (((x > 0) && pbFlag[y][x - 1] == 1) || (x < (nWidth-1) && pbFlag[y][x + 1] == 1) || (y>0 && pbFlag[y - 1][x] == 1) || (y< (nHeight-1) && pbFlag[y + 1][x] == 1))
		{
			if (pbFlag[y][x] == 1)
				pbFlag[y][x] = 0;

			//��������ߵĵ�
			if (((x > 0) && pbFlag[y][x - 1] == 1))
			{
				//��¼����Ĵ�С
				xStart = min(xStart, (x - 1));
				yStart = min(yStart, y);
				mystack.push(x - 1);
				mystack.push(y);
				x = x - 1;
				pbFlag[y][x] = 0;

			}

			//�������ұߵĵ�

			else if ((x < (nWidth-1)) && pbFlag[y][x + 1] == 1)
			{
				//��¼����Ĵ�С
				xEnd = max(xEnd, (x + 1));
				yStart = min(yStart, y);
				mystack.push(x + 1);
				mystack.push(y);
				x = x + 1;
				pbFlag[y][x] = 0;
			}

			//���������ϵĵ�

			else if (y > 0 && pbFlag[y - 1][x] == 1)
			{
				//��¼����Ĵ�С
				yStart = min(yStart, (y - 1));
				mystack.push(x);
				mystack.push(y - 1);
				y = y - 1;
				pbFlag[y][x] = 0;
			}

			// �������±ߵĵ� 

			else if (y < (nHeight-1) && pbFlag[y + 1][x] == 1)
			{
				// ��¼����Ĵ�С 
				yEnd = max(yEnd, (y + 1));
				mystack.push(x);
				mystack.push(y + 1);
				y = y + 1;
				pbFlag[y][x] = 0;
			}
			else{
				pbFlag[y][x] = 0;
			}
		}
	}
}

//���ο��������ɸѡ
void Searchalgorithm::rectangle_replan(int *size, int *sizenew)
{
	int sizetemp[4*MAXDETECTIONREGIONS];
	//�ж��Ƿ���ڰ�����ϵ,���������Ǹ�������ľ���ΪX_START��X_END��Y_START��Y_END��
	for (int i = 0; i < vaild_num; i++)
	{
		for (int j = i + 1; j < vaild_num; j++)
		{
			int judge = contain(size[i * 4 + 0], size[i * 4 + 1], size[i * 4 + 2], size[i * 4 + 3], size[j * 4 + 0], size[j * 4 + 1], size[j * 4 + 2], size[j * 4 + 3]);
			if (0 == judge)
			{
				size[j * 4 + 0] = X_START;
				size[j * 4 + 1] = X_END;
				size[j * 4 + 2] = Y_START;
				size[j * 4 + 3] = Y_END;
			}
			if (1 == judge)
			{
				size[i * 4 + 0] = X_START;
				size[i * 4 + 1] = X_END;
				size[i * 4 + 2] = Y_START;
				size[i * 4 + 3] = Y_END;
			}
		}
	}
	rectangleupdate(size, sizetemp);       //��������
}

//�жϾ����Ƿ���а�����ϵ
int Searchalgorithm::contain(int xs, int xe, int ys, int ye, int xs1, int xe1, int ys1, int ye1)
{
	int judge = 0;
	if (xs <= xs1 && ys <= ys1 && xe >= xe1 && ye >= ye1)
	{
		judge = 1;
	}
	if (xs >= xs1 && ys >= ys1 && xe <= xe1 && ye <= ye1)
	{
		judge = 2;
	}
	if (judge == 1){
		return 0;
	}
	else if (judge == 2){
		return 1;
	}else{
		return 2;
	}
}

//�����������
void Searchalgorithm::rectangleupdate(int *size, int *sizenew)
{
	int count_N = 0;
	for (int count = 0; count < vaild_num; count++)
	{
		if (size[count * 4] != X_START && size[count * 4 + 1] != X_END && size[count * 4 + 2] != Y_START && size[count * 4 + 3] != Y_END)
		{
			sizenew[count_N * 4 + 0] = size[count * 4];
			sizenew[count_N * 4 + 1] = size[count * 4 + 1];
			//if(sizenew[count_N * 4 + 0]==BLACKLINE_WIDTH)sizenew[count_N * 4 + 0]=0;
			//if(sizenew[count_N * 4 + 1]==nWidth-BLACKLINE_WIDTH-1)sizenew[count_N * 4 + 1] =nWidth-1;
					
			sizenew[count_N * 4 + 2] = size[count * 4 + 2];
			sizenew[count_N * 4 + 3] = size[count * 4 + 3];
			count_N++;
		}
		if(count_N>=MAXDETECTIONREGIONS)break;
	}
	vaild_num = count_N;
}
