// 导入青藏高原区域
var roi = ee.FeatureCollection("projects/ee-gzj/assets/QZGY_Qinghai-Tibet")
// var roi = ee.FeatureCollection("projects/ee-gzj/assets/Tibet_down")

var sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");

// Applies scaling factors.
function applyScaleFactors(image) {
  var opticalBands = image.select('B.').multiply(0.0000275).add(-0.2);
  // var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true);
              // .addBands(thermalBands, null, true);
}

function maskS2clouds(image) {
  var qa = image.select('QA60');
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}  //去云处理

var filteredImages = sentinel2
                      .filterBounds(roi)
                      .filterDate('2023-07-01', '2023-09-30') // 设置时间范围
                      // .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))// 选择云量少于10%的影像
                      // .map(applyScaleFactors)
                      .select('B2','B3','B4','B8','B11')
                      .sort('CLOUDY_PIXEL_PERCENTAGE', false)
                      .mosaic()
                      .clip(roi);

// 添加影像图层到地图上
// 可选：调整显示效果，例如通过visualization parameters
var visualization = {
  bands: ['B4', 'B3', 'B2'], // 对应于红绿蓝波段，适用于真彩色显示
  min: 0,
  max: 3000, // 根据实际情况调整最小和最大值以优化显示效果
};
var bands = filteredImages.bandNames().getInfo();
print(bands)

Map.addLayer(filteredImages, visualization, 'Sentinel-2 Image'); // 在地图上添加影像图层

// 设置地图中心点坐标和缩放级别
var longitude = 96.841; // 经度（西经为负值）
var latitude = 31.428; // 纬度
var zoomLevel = 15; // 缩放级别，数字越大，放大倍数越高

// 使用Map.setCenter()方法定位地图
Map.setCenter(longitude, latitude, zoomLevel);

//download
// 定义围绕兴趣点的小范围区域
var point = ee.Geometry.Point([longitude, latitude]); // 创建点 geometry
var bufferDistance = 0.1; // 设定缓冲区距离，单位为度，可根据需要调整大小
var roi1 = point.buffer(bufferDistance); // 使用缓冲区方法创建一个矩形区域


for (var i=0; i<filteredImages.bandNames().size().getInfo();i++){
  Export.image.toDrive({
    image: filteredImages.select(bands[i]),        //设置要输出的影像
    fileNamePrefix: bands[i],
    folder: "哨兵2",               //设置下载影像在Drive中存储的文件夹名称（可不设置）
    scale:10,    //空间分辨率，单位：米
    description: "sent2"+bands[i],    // 设置下载任务tasks的名称
    maxPixels:1e13,                       //单幅影像输出的最大像元数
    region:roi  //要下载影像的范围
  });
}
