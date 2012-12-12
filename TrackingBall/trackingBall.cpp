#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <cstdio>
#include <ctime>


using namespace cv;
using namespace std;

void skip(int pos, void * data)
{
  VideoCapture cap = *(VideoCapture *) data;
  if(pos != cap.get(CV_CAP_PROP_POS_FRAMES)+1)
    cap.set(CV_CAP_PROP_POS_FRAMES,(double) pos);
}


int diff3images(char*name)
{
    VideoCapture cap;
    cap.open(name);
    if(!cap.isOpened())  // check if we succeeded
      return -1;
    Mat edges;
    Mat edges2;
    Mat edges3;
    Mat diff;
    Mat diff2;
    cap >> edges;
    edges.copyTo(edges2);
    edges.copyTo(edges3);
    double w;
    double h;
    double f;
    double fc;
    int pos = 0;
    w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    f = cap.get(CV_CAP_PROP_FPS);
    fc = cap.get(CV_CAP_PROP_FRAME_COUNT);

    char chaine[11];
    sprintf(chaine,"%dx%d %d",(int)w,(int)h,(int)f);

    namedWindow("video",1);
    createTrackbar("bar","video", &pos,fc,skip,&cap);
    while(1)
      {
        setTrackbarPos("bar","video",getTrackbarPos("bar","video")+1);
        edges2.copyTo(edges3);
        edges.copyTo(edges2);
        cap >> edges;

        absdiff(edges3,edges2,diff2);
        absdiff(edges,edges2,diff);
        diff = diff&diff2;
        threshold(diff, diff, 30, 255,THRESH_BINARY);


        //putText(edges,chaine,Point(30,30),FONT_HERSHEY_COMPLEX_SMALL,1,{0,0,0});
        imshow("video", diff);
        if(waitKey(30) >= 0) break;
      }
    return 0;
}

int diff2images(char*name)
{
    VideoCapture cap;
      cap.open(name);
      if(!cap.isOpened())  // check if we succeeded
        return -1;
      Mat edges;
      Mat edges2;
      Mat diff;
      cap >> edges;
      edges.copyTo(edges2);
      double w;
      double h;
      double f;
      double fc;
      int pos = 0;
      w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
      h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
      f = cap.get(CV_CAP_PROP_FPS);
      fc = cap.get(CV_CAP_PROP_FRAME_COUNT);

      char chaine[11];
      sprintf(chaine,"%dx%d %d",(int)w,(int)h,(int)f);

      namedWindow("video2",1);
      createTrackbar("bar","video2", &pos,fc,skip,&cap);
      while(1)
        {
          setTrackbarPos("bar","video2",getTrackbarPos("bar","video2")+1);
          edges.copyTo(edges2);
          cap >> edges;
          absdiff(edges,edges2,diff);
          threshold(diff, diff, 30, 255,THRESH_BINARY);


          //putText(edges,chaine,Point(30,30),FONT_HERSHEY_COMPLEX_SMALL,1,{0,0,0});
          imshow("video2", diff);
          if(waitKey(30) >= 0) break;
        }
      return 0;
}

int background(char*name)
{
    VideoCapture cap;
      cap.open(name);
      if(!cap.isOpened())  // check if we succeeded
        return -1;
      Mat edges;
      Mat fgmask;
      Mat diff;
      BackgroundSubtractorMOG bg;
      cap >> edges;

      double fc;
      int pos = 0;
      fc = cap.get(CV_CAP_PROP_FRAME_COUNT);


      namedWindow("video2",1);
      createTrackbar("bar","video2", &pos,fc,skip,&cap);
      while(1)
        {
          setTrackbarPos("bar","video2",getTrackbarPos("bar","video2")+1);
          bg(edges,fgmask);
          cap >> edges;

          bg.getBackgroundImage(diff);
          threshold(fgmask, fgmask, 30, 255,THRESH_BINARY);

          imshow("video2", fgmask);
          if(waitKey(30) >= 0) break;
        }
      return 0;
}

int dilate_erode(char*name)
{
    VideoCapture cap;
      cap.open(name);
      if(!cap.isOpened())  // check if we succeeded
        return -1;
      Mat edges;
      Mat fgmask;
      Mat diff;
      vector<vector<Point> > contours;
      BackgroundSubtractorMOG bg;
      cap >> edges;

      double fc;
      int pos = 0;
      fc = cap.get(CV_CAP_PROP_FRAME_COUNT);


      namedWindow("video2",1);
      createTrackbar("bar","video2", &pos,fc,skip,&cap);
      while(1)
        {
          setTrackbarPos("bar","video2",getTrackbarPos("bar","video2")+1);
          Mat element = getStructuringElement( MORPH_RECT,Size( 3, 3 ),Point( 2, 2 ) );

          bg(edges,fgmask);
          cap >> edges;
          bg.getBackgroundImage(diff);
          threshold(fgmask, fgmask, 20, 255,THRESH_BINARY);

          //dilate(fgmask, fgmask,element );
          erode(fgmask, fgmask,element );
          dilate(fgmask, fgmask,element );

          vector<Vec4i> hierarchy;

          findContours(fgmask, contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_NONE,Point(0,0));
          drawContours(edges, contours,-1, Scalar( 0, 0,255),1, 8, hierarchy,1, Point(0,0));

          imshow("video2", edges);
          if(waitKey(30) >= 0) break;
        }
      return 0;

}

int ballTracking(char*name)
{
    VideoCapture cap;
      cap.open(name);
      if(!cap.isOpened())  // check if we succeeded
        return -1;
      Mat edges;
      Mat edges2;
      Mat diff;
      cap >> edges;

      double fc;
      int pos = 0;
      fc = cap.get(CV_CAP_PROP_FRAME_COUNT);


      namedWindow("video",1);
      namedWindow("video2",1);
      createTrackbar("bar","video", &pos,fc,skip,&cap);
      while(1)
        {
          setTrackbarPos("bar","video",getTrackbarPos("bar","video")+1);

          cap >> edges;
          edges.copyTo(edges2);

          for(int i=0;i<edges.rows;i++)
              for(int j=0;j<edges.cols;j++)
              {
                if(edges.at<Vec3b>(i,j)[2]<150 || edges.at<Vec3b>(i,j)[0]>50 || edges.at<Vec3b>(i,j)[1]>100)
                {
                    edges.at<Vec3b>(i,j)[0]=0;
                    edges.at<Vec3b>(i,j)[1]=0;
                    edges.at<Vec3b>(i,j)[2]=0;
                }
                else
                {

                        edges2.at<Vec3b>(i,j)[0]=0;
                        edges2.at<Vec3b>(i,j)[1]=0;
                        edges2.at<Vec3b>(i,j)[2]=255;
                }

              }
          /*vector<Vec3f> circles;
          cvtColor(edges, edges, CV_BGR2GRAY);
          HoughCircles(edges, circles, CV_HOUGH_GRADIENT, 1, 200, 100,100,0, 500 );

          for( size_t i = 0; i < circles.size(); i++ )
          {
               Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
               int radius = cvRound(circles[i][2]);
               // draw the circle center
               circle( edges, center, 3, Scalar(0,255,0), -1, 8, 0 );
               // draw the circle outline
               circle( edges, center, radius, Scalar(0,0,255), 3, 8, 0 );
          }
          //absdiff(edges,edges2,diff);

          cout << circles.size() << endl;*/

          Mat element = getStructuringElement( MORPH_RECT,Size( 3, 3 ),Point( 2, 2 ) );


          dilate(edges, edges,element );
          erode(edges, edges,element );
          dilate(edges, edges,element );
          dilate(edges, edges,element );

          imshow("video", edges);
          imshow("video2", edges2);
          if(waitKey(30) >= 0) break;
        }
      return 0;

}

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double scale, const Scalar& color)
{
for(int y = 0; y < cflowmap.rows; y += step)
    for(int x = 0; x < cflowmap.cols; x += step)
     {
        const Point2f& fxy = flow.at<Point2f>(y, x);
        line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
        circle(cflowmap, Point(x,y), 2, color, -1);
     }
}

int champdep(char*name)
{
    VideoCapture cap;
      cap.open(name);
      if(!cap.isOpened())  // check if we succeeded
        return -1;
      Mat edges;
      cap >> edges;
      Mat edges2;
      Mat edgesgrey;
      Mat edges2grey;
      edges.copyTo(edges2);
      Mat flow;
      edges.copyTo(flow);
      Mat flowmap;
      edges.copyTo(flowmap);


      double fc;
      int pos = 0;
      fc = cap.get(CV_CAP_PROP_FRAME_COUNT);


      namedWindow("video2",1);
      createTrackbar("bar","video2", &pos,fc,skip,&cap);
      while(1)
        {
          setTrackbarPos("bar","video2",getTrackbarPos("bar","video2")+1);
          edges.copyTo(edges2);
          cap >> edges;
          cvtColor(edges,edgesgrey,CV_RGB2GRAY);
          cvtColor(edges2,edges2grey,CV_RGB2GRAY);
          //cvtColor(flow,flow,CV_RGB2GRAY);
          calcOpticalFlowFarneback(edges2grey, edgesgrey, flow, 0.5, 2, 20, 1, 5, 1.1,0);

          drawOptFlowMap(flow, flowmap, 30,5, Scalar(255,255,255));
          imshow("video2", flowmap);
          if(waitKey(30) >= 0) break;
        }
      return 0;
}

int main(int argc, char*argv[])
{
    //diff3images(argv[1]);
    //diff2images(argv[1]);
    //background(argv[1]);
    //dilate_erode(argv[1]);
    //champdep(argv[1]);
    ballTracking(argv[1]);
    return 0;
}
