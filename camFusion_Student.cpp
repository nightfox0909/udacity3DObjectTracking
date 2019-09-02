
#include <iostream>
#include <algorithm>
#include <numeric>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    float averageEuclDistance = 0.0;
  for ( auto mt=kptMatches.begin(); mt!=kptMatches.end(); mt++ )
  {
    if(boundingBox.roi.contains(kptsCurr[mt->trainIdx].pt) && boundingBox.roi.contains(kptsPrev[mt->queryIdx].pt))
    {
      boundingBox.kptMatches.push_back(*mt);
    }
  }

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC,cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    float dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}

/*
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // prefilter lidar points to have minimum reflectivity
    LidarPoint d1=findMinX(lidarPointsCurr);
    LidarPoint d2=findMinX(lidarPointsPrev);
    cout<<d2.x<<" "<<d1.x<<endl;
    float dt=1/frameRate;

    //assume CV model
    if ((euclideanDistance(d2)-euclideanDistance(d1)) == 0)
    {
        TTC = NAN;
        return;
    }
    TTC=(euclideanDistance(d1)*dt)/(euclideanDistance(d2)-euclideanDistance(d1));

}*/

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate,double &TTC)
{
    // auxiliary variables
    double dT = 1/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
   std::vector<double> XPrevVect,XCurrVect;
   double meanCurr=0.0,meanPrev=0.0;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {       
        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
           XPrevVect.push_back(it->x);
            //minXPrev = minXPrev > it->x ? it->x : minXPrev;
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {

        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
          XCurrVect.push_back(it->x);
            //minXCurr = minXCurr > it->x ? it->x : minXCurr;
        }
    }
  
    std::sort(XPrevVect.begin(), XPrevVect.end());
    long medIndexPrev = floor(XPrevVect.size() / 2.0);
   double medPrevMinX = XPrevVect.size() % 2 == 0 ? (XPrevVect[medIndexPrev - 1] + XPrevVect[medIndexPrev]) / 2.0 : XPrevVect[medIndexPrev]; // compute median dist. ratio to remove outlier influence
  
   std::sort(XCurrVect.begin(), XCurrVect.end());
   long medIndexCurr = floor(XCurrVect.size() / 2.0);
   double medCurrMinX = XCurrVect.size() % 2 == 0 ? (XCurrVect[medIndexCurr - 1] + XCurrVect[medIndexCurr]) / 2.0 : XCurrVect[medIndexCurr]; // compute median dist. ratio to remove outlier influence

    // compute TTC from both measurements
    //TTC = minXCurr * dT / fabs(minXPrev - minXCurr);
    TTC=medCurrMinX * dT / fabs(medPrevMinX - medCurrMinX);
  	//cout<<"Normal TTC : "<<TTC<<" median TTC : "<< medCurrMinX * dT / fabs(medPrevMinX - medCurrMinX);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::vector<cv::KeyPoint> currMatchedkpts,prevMatchedkpts;
    
    for ( auto bb=currFrame.boundingBoxes.begin();bb!=currFrame.boundingBoxes.end();bb++)
    {
        std::vector<cv::DMatch> ROIdMatches;
        set<int> potentialMatches;
        std::vector<int> potentialMatchesNumber;
        //std::vector<std::pair<set<BoundingBox>,int>> potentialMatches;

        //Dmatches prev(queryidx),current(trainidx)
        for (auto mt=matches.begin();mt!=matches.end();mt++)
        {
            if(bb->roi.contains(currFrame.keypoints[mt->trainIdx].pt))
            {
                //ROIdMatches.push_back(*mt);
                for ( auto bb2=prevFrame.boundingBoxes.begin();bb2!=prevFrame.boundingBoxes.end();bb2++)
                {
                    if(bb2->roi.contains(prevFrame.keypoints[mt->queryIdx].pt)) 
                    {
                        auto tst = potentialMatches.insert(bb2->boxID);
                        if(tst.second)
                        {
                            potentialMatchesNumber.push_back(1);
                        }
                        else
                        {
                            //int findidx = potentialMatches.find(*bb2.id);
                            auto it = std::find(potentialMatches.begin(), potentialMatches.end(), bb2->boxID);
                            int index = std::distance(potentialMatches.begin(), it);
                            auto iter = potentialMatchesNumber.begin();
							std::advance(iter, index);
                           *iter+=1;
                        }
                        
                    }
                }
                if(potentialMatches.size()==1)
                {
                    bbBestMatches.insert(std::pair<int,int>(*potentialMatches.begin(),bb->boxID));
                }
                else if(potentialMatches.size()>1)
                {
                    auto maxElementIdx = std::max_element(potentialMatchesNumber.begin(),potentialMatchesNumber.end()) - potentialMatchesNumber.begin();
                    auto iter=potentialMatches.begin();
                    std::advance(iter,maxElementIdx);
                    bbBestMatches.insert(std::pair<int,int>(*iter,bb->boxID));
                }
                else
                {
                   bbBestMatches.insert(std::pair<int,int>(NULL,bb->boxID));
                }
            }
        }   

    }
}

/*
void searchBboxIntersection (BoundingBox &boundingBox,std::vector<cv::KeyPoint> &kpts, std::vector<cv::KeyPoint> &matchedkpts)
{
    //matchedkpts.erase();
    for (auto kpt=kpts.begin();kpt!=kpts.end();kpt++)
    {
        if(boundingBox.roi.contains(kpt->pt))
        {
            matchedkpts.push_back(*kpt);
        }

    }
}*/

LidarPoint findMinX (std::vector<LidarPoint> &lidarPoints)
{
    float min_distance=euclideanDistance(lidarPoints[0]);
    LidarPoint min=lidarPoints[0];

    for (auto lidrpt=lidarPoints.begin();lidrpt!=lidarPoints.end();lidrpt++)
    {
        if(min_distance < euclideanDistance(*lidrpt))
        {
            min=*lidrpt;
        }
    }
    return (min);
}

float euclideanDistance(LidarPoint pt1)
{
  LidarPoint pt2;
  pt2.x=0.0;
  pt2.y=0.0;
  pt2.z=0.0;
  pt2.r=0.0;
  return ( sqrt(  ((pt2.x-pt1.x)*(pt2.x-pt1.x)) + ((pt2.y-pt1.y)*(pt2.y-pt1.y)) + ((pt2.z-pt1.z)*(pt2.z-pt1.z)) ));
}  