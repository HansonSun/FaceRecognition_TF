conv0=slim.conv2d(64,[3,3],stride=1,scope="conv0")
bn0=slim.batch_norm(conv0)

stage1_unit1_bn1=slim.batch_norm(bn0)
stage1_unit1_conv1=slim.conv2d(stage1_unit1_bn1,64,[3,3],stride=1,scope="")
stage1_unit1_bn2=slim.batch_norm(stage1_unit1_conv1)
stage1_unit1_conv2=slim.conv2d(stage1_unit1_bn2,64,[3,3],stride=1,scope="stage1_unit1_conv2")
stage1_unit1_bn3=slim.batch_norm(stage1_unit1_conv2)

stage1_unit1_conv1sc=slim.conv2d(bn0,64,[1,1],stride=,scope="stage1_unit1_conv1sc")
stage1_unit1_sc=slim.batch_norm(stage1_unit1_conv1sc)

_plus0=tf.concat([stage1_unit1_bn3,stage1_unit1_sc],3)

stage1_unit2_bn1=slim.batch_norm(_plus0)
stage1_unit2_conv1=slim.conv2d(stage1_unit2_bn1,64,[3,3],stride=1,scope="stage1_unit2_conv1")
stage1_unit2_bn2=slim.batch_norm(stage1_unit2_conv1)
stage1_unit2_conv2=slim.conv2d(stage1_unit2_bn2,64,[3,3],stride=1,scope="stage1_unit2_conv2")
stage1_unit2_bn3=slim.batch_norm(stage1_unit2_conv2)


_plus1=tf.concat(_plus0,stage1_unit2_bn3)
stage1_unit3_bn1=slim.batch_norm(_plus1)
stage1_unit3_conv1=slim.conv2d(stage1_unit3_bn1,64,[3,3],stride=1,scope="stage1_unit3_conv1")
stage1_unit3_bn2=slim.batch_norm(stage1_unit3_conv1)
stage1_unit3_conv2=slim.conv2d(stage1_unit3_bn2,64,[3,3],tride=1,scope="stage1_unit3_conv2")
stage1_unit3_bn3=slim.batch_norm(stage1_unit3_conv2)

_plus2=tf.concat(_plus1,stage1_unit3_bn3)

stage2_unit1_bn1=slim.batch_norm(_plus2)
stage2_unit1_conv1=slim.conv2d(stage2_unit1_bn1,128,[3,3],stride=1,scope="stage2_unit1_conv1")
stage2_unit1_bn2=slim.batch_norm(stage2_unit1_conv1)
stage2_unit1_conv2=slim.conv2d(stage2_unit1_bn2,128,[3,3],stride=1,scope="stage2_unit1_conv2")
stage2_unit1_bn3=slim.batch_norm(stage2_unit1_conv2)

stage2_unit1_conv1sc=slim.conv2d(_plus2,128,[1,1],stride=1,scopre="stage2_unit1_conv1sc")
stage2_unit1_sc=slim.batch_norm(stage2_unit1_conv1sc)

_plus3=tf.concat([stage2_unit1_sc,stage2_unit1_bn3],3)

stage2_unit2_bn1=slim.batch_norm(_plus3)
stage2_unit2_conv1=slim.conv2d(,128,[3,3],1,stride=,scope="stage2_unit2_conv1")
stage2_unit2_bn2=slim.batch_norm(stage2_unit2_bn1)
stage2_unit2_conv2=slim.conv2d(,128,[3,3],1,stride=,scope="stage2_unit2_conv2")
stage2_unit2_bn3=slim.batch_norm(stage2_unit2_conv2)
_plus4=tf.concat([_plus3,stage2_unit2_bn3],3)

stage2_unit3_bn1=slim.batch_norm(_plus4)
stage2_unit3_conv1=slim.conv2d(,128,[3,3],1,stride=,scope="stage2_unit3_conv1")
stage2_unit3_bn2=slim.batch_norm(stage2_unit3_conv1)
stage2_unit3_conv2=slim.conv2d(,128,[3,3],1,stride=,scope="stage2_unit3_conv2")
stage2_unit3_bn3=slim.batch_norm(stage2_unit3_conv2)
_plus5=tf.concat([_plus4,stage2_unit3_bn3],3)

stage2_unit4_bn1=slim.batch_norm(_plus5)
stage2_unit4_conv1=slim.conv2d(,128,[3,3],1,stride=,scope="stage2_unit4_conv1")
stage2_unit4_bn2=slim.batch_norm(stage2_unit4_conv1)
stage2_unit4_conv2=slim.conv2d(,128,[3,3],1,stride=,scope="stage2_unit4_conv2")
stage2_unit4_bn3=slim.batch_norm(stage2_unit4_conv2)
_plus6=tf.concat([_plus5,stage2_unit4_bn3],3)

stage3_unit1_bn1=slim.batch_norm(_plus6)
stage3_unit1_conv1=slim.conv2d(,256,[3,3],1,stride=,scope="stage3_unit1_conv1")
stage3_unit1_bn2=slim.batch_norm(stage3_unit1_conv1)
stage3_unit1_conv2=slim.conv2d(,256,[3,3],1,stride=2,scope="stage3_unit1_conv2")
stage3_unit1_bn3=slim.batch_norm(stage3_unit1_conv2)

stage3_unit1_conv1_sc=slim.conv2d(,256,stride=2,scope="stage3_unit1_conv1_sc")
stage3_unit1_sc=slim.batch_norm()
_plus7=tf.concat([stage3_unit1_bn3,stage3_unit1_sc],3)

stage3_unit2_bn1=slim.batch_norm(_plus7)
stage3_unit2_conv1=slim.conv2d(stage3_unit2_bn1,256,[3,3],stride=1,scope="stage3_unit2_conv1")
stage3_unit2_bn2=slim.batch_norm(stage3_unit2_conv1)
stage3_unit2_conv2=slim.conv2d(stage3_unit2_bn2,256,[3,3],stride=1,scope="stage3_unit2_conv2")
stage3_unit2_bn3=slim.batch_norm(stage3_unit2_conv2)
_plus8=tf.concat(_plus7,stage3_unit2_bn3)

stage3_unit3_bn1=slim.batch_norm(_plus8)
stage3_unit3_conv1=slim.conv2d(stage3_unit3_bn1,256,[3,3],stride=1,scope="stage3_unit3_conv1")
stage3_unit3_bn2=slim.batch_norm(stage3_unit3_conv1)
stage3_unit3_conv2=slim.conv2d(stage3_unit3_bn2,256,[3,3],stride=1,scope="stage3_unit3_conv2")
stage3_unit3_bn3=slim.batch_norm(stage3_unit3_conv2)
_plus9=tf.concat([_plus8,stage3_unit3_bn3],3)

stage3_unit4_bn1=slim.batch_norm(_plus9)
stage3_unit4_conv1=slim.conv2d(stage3_unit4_bn1,256,[3,3],stride=1,scope="stage3_unit4_conv1")
stage3_unit4_bn2=slim.batch_norm(stage3_unit4_conv1)
stage3_unit4_conv2=slim.conv2d(stage3_unit4_bn2,256,[3,3],stride=1,scope="stage3_unit4_conv2")
stage3_unit4_bn3=slim.batch_norm(stage3_unit4_conv2)
_plus10=tf.concat([_plus9,stage3_unit4_bn3],3)

stage3_unit5_bn1=slim.batch_norm(_plus10)
stage3_unit5_conv1=slim.conv2d(stage3_unit5_bn1,256,[3,3,]stride=1,scope="stage3_unit5_conv1")
stage3_unit5_conv1=slim.batch_norm(stage3_unit5_conv1)
stage3_unit5_conv2=slim.conv2d(stage3_unit5_conv1,256,[3,3],stride=1,scope="stage3_unit5_conv2")
stage3_unit5_conv2=slim.batch_norm(stage3_unit5_conv2)
_plus11=tf.concat([_plus10,stage3_unit5_conv2],3)

stage3_unit6_bn1=slim.batch_norm(_plus11)
stage3_unit6_conv1=slim.conv2d(stage3_unit6_bn1,256,[3,3],stride=1,scope="stage3_unit6_conv1")
stage3_unit6_conv1=slim.batch_norm(stage3_unit6_conv1)
stage3_unit6_conv2=slim.conv2d(stage3_unit6_conv1,256,[3,3],stride=1,scope="stage3_unit6_conv2")
stage3_unit6_conv2=slim.batch_norm(stage3_unit6_conv2)
_plus12=tf.concat([_plus11,stage3_unit6_conv2],3)

stage4_unit1_bn1=slim.batch_norm(_plus12)
stage4_unit1_conv1=slim.conv2d(stage4_unit1_bn1,512,[3,3],stride=1,scope="stage4_unit1_conv1")
stage4_unit1_bn2=slim.batch_norm(stage4_unit1_conv1)
stage4_unit1_conv2=slim.conv2d(stage4_unit1_bn2,512,[3,3],stride=2,scope="stage4_unit1_conv2")
stage4_unit1_bn3=slim.batch_norm(stage4_unit1_conv2)

stage4_unit1_conv1sc=slim.conv2d(,512,[3,3],stride=2,scope="stage4_unit1_conv1sc")
stage4_unit1_sc=slim.batch_norm(stage4_unit1_conv1sc)
_plus13=tf.concat([stage4_unit1_bn3,stage4_unit1_sc],3)

stage4_unit2_bn1=slim.batch_norm(_plus13)
stage4_unit2_conv1=slim.conv2d(stage4_unit2_bn1,512,[3,3],stride=1,scope="stage4_unit2_conv1")
stage4_unit2_bn2=slim.batch_norm(stage4_unit2_conv1)
stage4_unit2_conv2=slim.conv2d(stage4_unit2_bn2,512,[3,3],stride=1,scope="stage4_unit2_conv2")
stage4_unit2_bn3=slim.batch_norm(stage4_unit2_conv2)
_plus14=tf.concat([_plus13,stage4_unit2_bn3],3)

stage4_unit3_bn1=slim.batch_norm(_plus14)
stage4_unit3_conv1=slim.conv2d(stage4_unit3_bn1,512,[3,3],stride=1,scope="stage4_unit3_conv1")
stage4_unit3_bn2=slim.batch_norm(stage4_unit3_conv1)
stage4_unit3_conv2=slim.conv2d(stage4_unit3_bn2,512,[3,3],stride=1,scope="stage4_unit3_conv2")
stage4_unit3_bn3=slim.batch_norm(stage4_unit3_conv2)
_plus15=tf.concat([_plus14,stage4_unit3_bn3],4)