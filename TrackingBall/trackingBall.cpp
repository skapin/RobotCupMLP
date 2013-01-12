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



      KalmanFilter KF(4, 2, 0);
      KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
      Mat_<float> measurement(2,1);
      measurement.setTo(Scalar(0));

      // init...
      KF.statePre.at<float>(0) = 0;
      KF.statePre.at<float>(1) = 0;
      KF.statePre.at<float>(2) = 0;
      KF.statePre.at<float>(3) = 0;
      setIdentity(KF.measurementMatrix);
      setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
      setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
      setIdentity(KF.errorCovPost, Scalar::all(.1));


      //createTrackbar("bar","video", &pos,fc,skip,&cap);
      while(1)
        {
          //setTrackbarPos("bar","video",getTrackbarPos("bar","video")+1);

          cap >> edges;
          edges.copyTo(edges2);
          Point bary;
          bary.x = 0;
          bary.y = 0;
          Point bary2;
          bary2.x = 0;
          bary2.y = 0;
          int n = 0;
          double x = 0;
          double y = 0;
          int n2 = 0;
          double x2 = 0;
          double y2 = 0;


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

                        //edges2.at<Vec3b>(i,j)[0]=0;
                        //edges2.at<Vec3b>(i,j)[1]=0;
                        //edges2.at<Vec3b>(i,j)[2]=255;
                    if(std::abs(j-edges.cols/4) < std::abs(j-3*edges.cols/4))
                    {
                        n++;
                        x += j;
                        y += i;
                    }
                    else
                    {
                        n2++;
                        x2 += j;
                        y2 += i;
                    }
                }

             }
          x = x/n;
          y = y/n;
          x2 = x2/n2;
          y2 = y2/n2;

          if(std::abs(x-x2) < 15 && std::abs(y-y2) <15)
          {
              x = x*n + x2*n2;
              y = y*n +y2*n2;
              x = x/(n+n2);
              y = y/(n+n2);
              if(bary.x >= 20 && bary.y >=20)
                circle(edges2, bary, 15, Scalar(0,0,255), 2);
          }
          else
          {
          bary.x = x;
          bary.y = y;

          bary2.x = x2;
          bary2.y = y2;
          if(bary.x >= 20 && bary.y >=20 && n>25)
            circle(edges2, bary, 15, Scalar(0,0,255), 2);
          if(bary2.x >= 20 && bary2.y >=20 && n2>25)
            circle(edges2, bary2, 15, Scalar(0,0,255), 2);
           }
          //Kalman filter test http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/





          // First predict, to update the internal statePre variable
          Mat prediction = KF.predict();
          Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

          // Get mouse point
          measurement(0) = bary.x;
          measurement(1) = bary.y;

          Point measPt(measurement(0),measurement(1));

          // The "correct" phase that is going to use the predicted value and our measurement
          Mat estimated = KF.correct(measurement);
          Point statePt(estimated.at<float>(0),estimated.at<float>(1));
            
          circle(edges, statePt, 2, Scalar(255,255,255), -1);
          //kalman filter end


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
