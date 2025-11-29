// 导入青藏高原区域
// var roi = ee.FeatureCollection("projects/ee-gzj/assets/QZGY_Qinghai-Tibet")
var roi = ee.FeatureCollection("projects/ee-gzj/assets/Tibet_down")

// var roi_geometry = roi.geometry();
// var dataset = ee.ImageCollection('COPERNICUS/DEM/GLO30')
//     .filterBounds(roi)   ;
// var elevation = dataset.select('DEM');
var SRTM30 = ee.ImageCollection('COPERNICUS/DEM/GLO30');

var elevation = SRTM30.select('DEM').mosaic()
                .clip(roi)

var slope = ee.Terrain.slope(elevation)

var cuttingRegion = roi.geometry();

var minMax = elevation.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry:roi,
  scale: 30,
  maxPixels: 1e13
});
var min = ee.Dictionary(minMax.get('min'))
var max = ee.Dictionary(minMax.get('max'))

Export.image.toDrive({

  image: elevation,

  description: 'COPERNICUS',

  scale: 30,

  maxPixels: 1e13,

  region: cuttingRegion });

// Export.image.toDrive({

//   image: slope,

//   description: 'FH_Slope',

//   scale: 30,

//   maxPixels: 1e13,

//   region: cuttingRegion });

 // 可视化参数
 var args = {

   crs: 'EPSG:3857',

   dimensions: '300',

   region: roi,

   min: -2000,

   max: 10000,

   palette: 'green, blanchedalmond,orange,black ',

   framesPerSecond: 12,

 };

Map.addLayer(roi,{},'roi_Boundary');

Map.centerObject(roi, 7);

// Map.addLayer(slope,{},'slope');

Map.addLayer(elevation,args,'elevation');
