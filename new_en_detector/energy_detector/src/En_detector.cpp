#include "En_detector.hpp"

namespace rm_auto_aim
{
  std::vector<Leaf> En_Detector::detect(const cv::Mat &input)
  {
    result_img = input.clone();
    auto leafs = work(result_img);
    leafs_ = Leaf_filter(leafs, input.cols, input.rows);
    R_Point = findR(input);
    return leafs_;
  }

  En_Detector::En_Detector(float NMS_THRESHOLD, float CONF_THRESHOLD, int detect_color, int binary_thres) : NMS_THRESHOLD(NMS_THRESHOLD), CONF_THRESHOLD(CONF_THRESHOLD), detect_color(detect_color), binary_thres(binary_thres)
  {
    model = core.read_model(MODEL_PATH);
    compiled_model = core.compile_model(model, DEVICE);
    infer_request = compiled_model.create_infer_request();
    input_port = compiled_model.input();
    kernel_3_3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    kernel_5_5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    kernel_7_7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
  }

  cv::Mat En_Detector::letter_box(cv::Mat &src)
  {
    int col = src.cols;
    int row = src.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    src.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
  }

  std::vector<Leaf> En_Detector::work(cv::Mat src_img)
  {
    this->debug_leafs.data.clear();
    this->leafs_.clear();
    int img_h = IMG_SIZE;
    int img_w = IMG_SIZE;

    // Preprocess the image
    cv::Mat boxed = letter_box(src_img);
    float scale = boxed.size[0] / IMG_SIZE;
    cv::Mat blob = cv::dnn::blobFromImage(boxed, 1.0 / 255.0, cv::Size(IMG_SIZE, IMG_SIZE), cv::Scalar(), true);
    //  Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    //  Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);
    // infer_request.infer();
    infer_request.start_async();
    infer_request.wait();
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();
    // Postprocess the result
    float *data = output.data<float>();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    cv::transpose(output_buffer, output_buffer); //[8400,13]
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> objects_keypoints;

    // box[cx, cy, w, h] + Score + [4,2] keypoints
    for (int i = 0; i < output_buffer.rows; i++)
    {
      float class_score = output_buffer.at<float>(i, 4);
      if (class_score < CONF_THRESHOLD)
        continue;
      else
      {
        class_scores.emplace_back(class_score);
        class_ids.emplace_back(0); //{0:"leaf"}
        float cx = output_buffer.at<float>(i, 0);
        float cy = output_buffer.at<float>(i, 1);
        float w = output_buffer.at<float>(i, 2);
        float h = output_buffer.at<float>(i, 3);

        // Get the box
        int left = int((cx - 0.5 * w) * scale);
        int top = int((cy - 0.5 * h) * scale);
        int width = int(w * scale);
        int height = int(h * scale);

        // Get the keypoints
        std::vector<cv::Point2f> keypoints;
        cv::Mat kpts = output_buffer.row(i).colRange(5, 13);
        for (int j = 0; j < KPT_NUM; j++)
        {
          float x = kpts.at<float>(0, j * 2 + 0) * scale;
          float y = kpts.at<float>(0, j * 2 + 1) * scale;
          cv::Point2f kpt(x, y);
          keypoints.emplace_back(kpt);
        }
        boxes.emplace_back(cv::Rect(left, top, width, height));
        objects_keypoints.emplace_back(keypoints);
      }
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, class_scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    // -------- Select the detection best result -----------
    for (size_t i = 0; i < indices.size(); i++)
    {
      int index = indices[i]; // best result index
      Leaf leaf;
      leaf.rect = boxes[index];
      leaf.label = class_ids[index];
      leaf.prob = class_scores[index];

      std::vector<cv::Point2f> object_keypoints = objects_keypoints[index];
      for (int i = 0; i < KPT_NUM; i++)
      {
        int x = std::clamp(int(object_keypoints[i].x), 0, src_img.cols);
        int y = std::clamp(int(object_keypoints[i].y), 0, src_img.rows);
        // Draw point
        leaf.kpt.emplace_back(cv::Point2f(x, y));
      }
      leafs_.emplace_back(leaf);
    }
    return leafs_;
  }

  std::vector<Leaf> En_Detector::Leaf_filter(
      std::vector<Leaf> &leafs, const int MAX_WIDTH, const int MAX_HEIGHT)
  {
    float angle;
    auto Get_Point = [&](cv::Point2f pt1, cv::Point2f pt2) -> std::vector<cv::Point2f>
    {
      std::vector<cv::Point2f> Point_2;
      cv::Point2f center = (pt1 + pt2) * 0.5f;
      float width = std::abs(pt2.x - pt1.x);
      float height = std::abs(pt2.y - pt1.y);
      angle = std::atan2(pt2.y - pt1.y, pt2.x - pt1.x);
      angle = angle * 180.0f / CV_PI;
      // //std::cout<<"angle"<<angle<<std::endl;

      cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1);
      std::vector<cv::Point2f> points;
      points.emplace_back(pt1);
      points.emplace_back(pt2);

      // 逆向旋转已知的两个对角点
      std::vector<cv::Point2f> rotatedPoints;
      cv::transform(points, rotatedPoints, rotationMatrix);
      Point_2.emplace_back(rotatedPoints[0]);
      Point_2.emplace_back(rotatedPoints[1]);
      return Point_2;
    };

    std::vector<Leaf> result;
    for (auto &leaf : leafs)
    {
      float x0 = leaf.rect.x;
      float y0 = leaf.rect.y;
      float x1 = leaf.rect.x + leaf.rect.width;
      float y1 = leaf.rect.y + leaf.rect.height;
      cv::Point2f keypoints_center(0, 0);
      std::vector<bool> valid_keypoints(4, false);
      for (int i = 0; i < leaf.kpt.size(); i++)
      {
        if (leaf.kpt[i].x != 0 && leaf.kpt[i].y != 0)
        {
          valid_keypoints[i] = true;
        }
      }
      std::vector<cv::Point2f> pair_point;
      // 四种情况判断
      if (valid_keypoints[0] && valid_keypoints[1] && valid_keypoints[2] && valid_keypoints[3])
      {
        // 1. 四个关键点都有效，直接取中心点im
        keypoints_center = (leaf.kpt[0] + leaf.kpt[1] + leaf.kpt[3] + leaf.kpt[2]) * 0.25;
      }
      else if (
          valid_keypoints[0] && valid_keypoints[2] && (!valid_keypoints[1] || !valid_keypoints[3]))
      {
        // 2. 0 2关键点有效，1 3 关键点缺少一个以上： 算 0 2 关键点的中点
        keypoints_center = (leaf.kpt[0] + leaf.kpt[2]) * 0.5;
        pair_point = Get_Point(leaf.kpt[0], leaf.kpt[2]);
        if (angle > 0 && angle < 180)
        {
          leaf.kpt[1] = valid_keypoints[1] ? leaf.kpt[1] : pair_point.at(0);
          leaf.kpt[3] = valid_keypoints[3] ? leaf.kpt[3] : pair_point.at(1);
        }
        else
        {
          leaf.kpt[1] = valid_keypoints[1] ? leaf.kpt[1] : pair_point.at(1);
          leaf.kpt[3] = valid_keypoints[3] ? leaf.kpt[3] : pair_point.at(0);
        }
      }
      else if (
          valid_keypoints[1] && valid_keypoints[3] && (!valid_keypoints[0] || !valid_keypoints[2]))
      {
        // 3. 1 3关键点有效，0 2 关键点缺少一个以上： 算 1 3 关键点的中点
        keypoints_center = (leaf.kpt[1] + leaf.kpt[3]) * 0.5;
        pair_point = Get_Point(leaf.kpt[1], leaf.kpt[3]);
        if (angle > 0 && angle < 180)
        {
          leaf.kpt[0] = valid_keypoints[0] ? leaf.kpt[0] : pair_point.at(0);
          leaf.kpt[2] = valid_keypoints[2] ? leaf.kpt[2] : pair_point.at(1);
        }
        else
        {
          leaf.kpt[0] = valid_keypoints[0] ? leaf.kpt[0] : pair_point.at(1);
          leaf.kpt[2] = valid_keypoints[2] ? leaf.kpt[2] : pair_point.at(0);
        }
      }
      else
      {
        // 4. 以上三个都不满足，算bbox中心点
        keypoints_center = cv::Point2f(x0 + leaf.rect.width / 2, y0 + leaf.rect.height / 2);
      }
      leaf.kpt.emplace_back(keypoints_center);
      leaf.leaf_type = true;
      for (size_t i = 0; i < leaf.kpt.size(); i++)
      {
        if (
            leaf.kpt[i].x < 0 or leaf.kpt[i].x > MAX_WIDTH or leaf.kpt[i].y < 0 or
            leaf.kpt[i].x > MAX_HEIGHT or leaf.kpt[i].x < 0 or leaf.kpt[i].x > MAX_WIDTH or
            leaf.kpt[i].y < 0 or leaf.kpt[i].y > MAX_HEIGHT or R_Point.x <= 0 or R_Point.y <= 0)
        {

          leaf.leaf_type = false;
          break;
        }
      }
      result.emplace_back(leaf);
    }
    return result;
  }
  void En_Detector::drawRuselt(cv::Mat &src)
  {
    for (auto &leaf : leafs_)
    {
      float x0 = leaf.rect.x;
      float y0 = leaf.rect.y;
      float x1 = leaf.rect.x + leaf.rect.width;
      float y1 = leaf.rect.y + leaf.rect.height;
      int baseLine;
      float prob = leaf.prob;
      cv::rectangle(src, leaf.rect, cv::Scalar(0, 0, 255), 2, 8);
      std::string label = class_names[leaf.label] + std::to_string(leaf.prob).substr(0, 4);
      cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
      cv::Rect textBox(leaf.rect.tl().x, leaf.rect.tl().y - 15, textSize.width, textSize.height + 5);
      cv::rectangle(src, textBox, cv::Scalar(0, 0, 255), cv::FILLED);
      cv::putText(src, label, cv::Point(leaf.rect.tl().x, leaf.rect.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
      for (auto point : leaf.kpt)
      {
        cv::circle(src, point, 5, cv::Scalar(255, 0, 255), -1);
      }
      cv::circle(src, R_Point, 5, cv::Scalar(255, 0, 255), -1);
    }
  }
  cv::Point2f En_Detector::findR(cv::Mat src)
  {
    if(leafs_.empty())return cv::Point2f(0,0);
    GaussianBlur(src, src, cv::Size(5, 5), 1.0);
    cv::Point2f R_, result;
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    this->detect_color == RED ? cv::threshold(channels[2] * 1.2 - channels[0], bin, binary_thres, 255, cv::THRESH_BINARY) : cv::threshold(channels[0] * 0.9 - channels[2], bin, 100, 255, cv::THRESH_BINARY);
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel_3_3, cv::Point(-1, -1));
    cv::dilate(bin, bin, kernel_3_3);
    cv::dilate(bin, bin, kernel_5_5);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hiearachy_test;
    cv::findContours(bin, contours, hiearachy_test, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.size() > 0)
    {
      for (size_t i = 0; i < contours.size(); i++)
      {
        if (cv::contourArea(contours[i]) >= 150&&contourArea(contours[i])<=650)
        {
          cv::Rect rect = boundingRect(contours[i]);
          cv::Mat ROI = cv::Mat(bin, rect);
          cv::resize(ROI, ROI, cv::Size(50, 50));
          std::vector<std::vector<cv::Point>> contours_;
          std::vector<cv::Vec4i> hiearachy_test_;
          cv::findContours(ROI, contours_, hiearachy_test_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
          R_ = cv::Point2f(rect.x + rect.width / 2, rect.y + rect.height / 2);
          
          float x = leafs_[0].kpt[4].x - R_.x, y = leafs_[0].kpt[4].y - R_.y;
          float tmp = sqrt(x * x + y * y);
          if (tmp > 80)
          {
            result = R_;
          }
        }
      }
    }
    return result;
  }
} // namespace rm_auto_aim
