// Filter by metadata properties.
var sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD');
var vh = sentinel1
  // Filter to get images with VV and VH dual polarization.
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  // Filter to get images collected in interferometric wide swath mode.
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
 .map(function(image) {
          var edge = image.lt(-30.0);
          var maskedImage = image.mask().and(edge.not());
          return image.updateMask(maskedImage).clip(roi);
        });

// Filter to get images from different look angles.
var vhAscending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));
var vhDescending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
var spring = ee.Filter.date('2023-07-01', '2023-09-30');
// Create a composite from means at different polarizations and look angles.
var composite = ee.Image.cat([
  // vhAscending.select('VH').mean(),
  ee.ImageCollection(vhAscending.select('VH').merge(vhDescending.select('VH'))).filter(spring).mean(),
  ee.ImageCollection(vhAscending.select('VV').merge(vhDescending.select('VV'))).filter(spring).mean(),
  // vhDescending.select('VH').mean()
]).focal_median();
print('composite',composite.bandNames().getInfo());
// Display as a composite of polarization and backscattering characteristics.
Map.addLayer(composite, {min: -20, max: 0}, 'composite');


var bands = composite.bandNames().getInfo();

for (var i=0; i<composite.bandNames().size().getInfo();i++){
  Export.image.toDrive({
    image: composite.select(bands[i]),        //设置要输出的影像
    fileNamePrefix: bands[i],
    folder: "哨兵1",               //设置下载影像在Drive中存储的文件夹名称（可不设置）
    scale:10,    //空间分辨率，单位：米
    description: "sent1"+bands[i],    // 设置下载任务tasks的名称
    maxPixels:1e13,                       //单幅影像输出的最大像元数
    region:roi  //要下载影像的范围
  });
}