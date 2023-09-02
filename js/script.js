var canvas = document.getElementById("canvas");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// 初始化
var gl = canvas.getContext("webgl");
if (!gl) {
	console.error("Unable to initialize WebGL.");
}

var time = 0.0;

//顶点着色器
var vertexSource = `
// 名为 position 的属性，类型为 vec2 ，二维向量，用于表示顶点的位置信息
// 在顶点着色器中，我们可以使用这个属性来操作顶点的位置。
attribute vec2 position;
void main() {
	// 向量构造函数，用于创建一个四维向量
	gl_Position = vec4(position, 0.0, 1.0);// position 向量的 x 和 y 分量作为前两个参数，然后将 0.0 和 1.0 作为后两个参数。此四维向量表示一个顶点的位置信息，其中 x 和 y 分量来自 position，而 z 和 w 分量分别被设置为 0.0 和 1.0。
}
`;
//片元着色器
var fragmentSource = `
// 设定了默认精度
precision highp float; 

// uniform 全局只读
uniform float width; 
uniform float height;

// 二维浮点数向量
vec2 resolution = vec2(width, height); 

uniform float time;

// 预编译指令 points[数组长度]
#define POINT_COUNT 8 

// 创建了一个名为 points 的数组，数组的元素类型是 vec2。vec2 表示一个二维向量，通常用于存储坐标或位置信息。
vec2 points[POINT_COUNT];
const float speed = -0.5;
const float len = 0.25;
float intensity = 1.3;
float radius = 0.008;

//Signed distance to a quadratic bezier 到二次贝塞尔曲线的带符号距离
float sdBezier(vec2 pos, vec2 A, vec2 B, vec2 C){  
	// 二维向量 a = 向量 B - 向量 A  
	vec2 a = B - A;
	vec2 b = A - 2.0*B + C;
	vec2 c = a * 2.0;
	vec2 d = A - pos;

	// 标量 kk = 向量 b 与自身的点乘的倒数
	// 一维矩阵乘积：每个对应的点进行相乘，然后相加 [2,3,4] x [1,2,4] = 2*1 + 3*2 + 4*4 = 24
	float kk = 1.0 / dot(b,b);
	float kx = kk * dot(a,b);
	float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
	float kz = kk * dot(d,a);      

	float res = 0.0;

	float p = ky - kx*kx;
	float p3 = p*p*p;
	float q = kx*(2.0*kx*kx - 3.0*ky) + kz;
	float h = q*q + 4.0*p3;

	if(h >= 0.0){ 
		// √h
		h = sqrt(h);
		vec2 x = (vec2(h, -h) - q) / 2.0;
		// 向量 x 中每个分量的绝对值，对每个分量进行1/3次幂的计算，乘以 x 中每个分量的正负号
		vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
		// 将向量 uv 的两个分量相加，然后减去变量 kx 的值
		float t = uv.x + uv.y - kx;
		// 将 t 的值限制在区间 [0.0, 1.0] 内
		t = clamp( t, 0.0, 1.0 );

		vec2 qos = d + (c + b*t)*t;
		// 计算了向量 qos 的长度（模）
		res = length(qos);
	}else{
		// √-p
		float z = sqrt(-p);
		// 反余弦值
		float v = acos( q/(p*z*2.0) ) / 3.0;
		float m = cos(v);
		// √3
		float n = sin(v)*1.732050808;
		vec3 t = vec3(m + m, -n - m, n - m) * z - kx;
		t = clamp( t, 0.0, 1.0 );

		// 使用三个不同的根（t.x，t.y，t.z）来计算 qos 向量，并计算其内积 dis
		vec2 qos = d + (c + b*t.x)*t.x;
		float dis = dot(qos,qos);
        
		res = dis;

		qos = d + (c + b*t.y)*t.y;
		dis = dot(qos,qos);
		res = min(res,dis);
		
		qos = d + (c + b*t.z)*t.z;
		dis = dot(qos,qos);
		// min()返回最小值
		res = min(res,dis);

		res = sqrt( res );
	}
    
	return res;
}

// 计算心形曲线上某个时间值（t）对应的位置坐标的函数
vec2 getHeartPosition(float t){
	return vec2(16.0 * sin(t) * sin(t) * sin(t),
							-(13.0 * cos(t) - 5.0 * cos(2.0*t)
							  - 2.0 * cos(3.0*t) - cos(4.0*t)));
}

// 计算光晕效果的函数  dist 是光源到某点的距离，radius 是光源的半径，intensity 是强度参数
// 当距离越接近半径时，表达式的值会越接近 1，表示光晕的强度较高；
// 当距离 dist 较远时，表达式的值会越接近 0，表示光晕的强度较低。
float getGlow(float dist, float radius, float intensity){
	return pow(radius/dist, intensity);
}

// 计算曲线上某一点到给定点的最短距离
// t，时间；pos，给定点的坐标；offset，偏移量；scale，比例因子
float getSegment(float t, vec2 pos, float offset, float scale){
	for(int i = 0; i < POINT_COUNT; i++){
		// 点的位置 存储在 points 的数组中
		points[i] = getHeartPosition(offset + float(i)*len + fract(speed * t) * 6.28);
	}
    
	// 中点
	vec2 c = (points[0] + points[1]) / 2.0;
	// 存储前一个中点的值
	vec2 c_prev;
	// 初始距离
	float dist = 10000.0;
    
	// 计算给定点到曲线的最短距离
	for(int i = 0; i < POINT_COUNT-1; i++){
		// 中点c保存到c_prev中，以备下一次迭代使用
		c_prev = c;
		// 计算新的中点c，它是数组中相邻两个点的中点
		c = (points[i] + points[i+1]) / 2.0;
		// 计算给定点到曲线的最短距离，并将其与之前的最短距离dist进行比较，保留较小的那个距离
		dist = min(dist, sdBezier(pos, scale * c_prev, scale * points[i], scale * c));
	}
	// 返回最短距离dist 如果计算出的最短距离小于0，将返回0
	return max(0.0, dist);
}


// 纹理UV坐标
// 纹理坐标：一张纹理贴图图像的坐标，选择一张图片，比如以图片左下角为坐标原点，右上角为坐标(1.0,1.0)，图片上所有位置纵横坐标都介于0.0~1.0之间
void main(){
	// 计算片段的纹理坐标（uv坐标）
	// gl_FragCoord ：当前片段的像素坐标，resolution.xy ：渲染目标的宽度和高度
	// 得到了当前片段在纹理上的归一化坐标（范围在0到1之间），存储在变量 uv 中。
	vec2 uv = gl_FragCoord.xy/resolution.xy;

	// 渲染目标的宽高比 = 目标的宽度 / 目标的高度
	float widthHeightRatio = resolution.x/resolution.y;

	// 二维向量 centre，表示屏幕中心的归一化坐标被设置为 (0.5, 0.5)
	vec2 centre = vec2(0.5, 0.5);

	// 从屏幕中心到当前片段的向量
	vec2 pos = centre - uv;

	// 调整 pos 向量的纵坐标
	pos.y /= widthHeightRatio;
	//Shift upwards to centre heart 向上移动到心的中心
	pos.y += 0.02;
	float scale = 0.000015 * height;
	
	float t = time;
    
	//Get first segment 获取第一段
	// 最短距离
  float dist = getSegment(t, pos, 0.0, scale);
	// 光晕效果
  float glow = getGlow(dist, radius, intensity);
  
	// 三分量向量 初始化 (0.0,0.0,0.0) 黑色 存储颜色信息
  vec3 col = vec3(0.0);

	//White core 白色核心
	// smoothstep() 可以用来生成0到1的平滑过渡值，它也叫平滑阶梯函数
  col += 10.0*vec3(smoothstep(0.003, 0.001, dist));
  //Pink glow 粉色光芒
  col += glow * vec3(1.0,0.05,0.3);
  
  //Get second segment 获取第二段
  dist = getSegment(t, pos, 3.4, scale);
  glow = getGlow(dist, radius, intensity);
  
  //White core
  col += 10.0*vec3(smoothstep(0.003, 0.001, dist));
  //Blue glow
  col += glow * vec3(0.1,0.4,1.0);
        
	//Tone mapping 色调映射 exp() 以自然常数e为底的指数函数
	col = 1.0 - exp(-col);

	//Gamma 希腊字母表的第3个字母 γ
	col = pow(col, vec3(0.4545));

	//Output to screen 输出到屏幕
	// 四分量向量 RGB值 + alpha通道 控制颜色的透明度
 	gl_FragColor = vec4(col,1.0);
}
`;

//************** Utility functions **************

// type = resize 当浏览器窗口被调整大小时触发
// listener 会在该类型的事件捕获阶段传播到该 EventTarget 时触发。
window.addEventListener("resize", onWindowResize, false);

// listener
function onWindowResize() {
	canvas.width = window.innerWidth;
	canvas.height = window.innerHeight;
	gl.viewport(0, 0, canvas.width, canvas.height);
	gl.uniform1f(widthHandle, window.innerWidth);
	gl.uniform1f(heightHandle, window.innerHeight);
}

//Compile shader and combine with source 编译着色器 并与源代码合并
function compileShader(shaderSource, shaderType) {
	var shader = gl.createShader(shaderType);
	gl.shaderSource(shader, shaderSource);
	gl.compileShader(shader);
	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		throw "Shader compile failed with: " + gl.getShaderInfoLog(shader);
	}
	return shader;
}

//Utility to complain loudly if we fail to find the attribute/uniform
function getAttribLocation(program, name) {
	var attributeLocation = gl.getAttribLocation(program, name);
	if (attributeLocation === -1) {
		throw "Cannot find attribute " + name + ".";
	}
	return attributeLocation;
}

function getUniformLocation(program, name) {
	var attributeLocation = gl.getUniformLocation(program, name);
	if (attributeLocation === -1) {
		throw "Cannot find uniform " + name + ".";
	}
	return attributeLocation;
}

//************** Create shaders **************

//Create vertex and fragment shaders 创建顶点和片段着色器
var vertexShader = compileShader(vertexSource, gl.VERTEX_SHADER);
var fragmentShader = compileShader(fragmentSource, gl.FRAGMENT_SHADER);

//Create shader programs 创建着色器程序
var program = gl.createProgram();
gl.attachShader(program, vertexShader);
gl.attachShader(program, fragmentShader);
gl.linkProgram(program);

gl.useProgram(program);

//Set up rectangle covering entire canvas 设置覆盖整个画布的矩形
// vertexData 是一个包含了8个元素的 Float32Array 数组 表示了一个二维平面上的四个点的坐标。每个点的坐标由两个浮点数组成，分别表示X和Y坐标
var vertexData = new Float32Array([
	-1.0,
	1.0, // top left
	-1.0,
	-1.0, // bottom left
	1.0,
	1.0, // top right
	1.0,
	-1.0, // bottom right
]);

//Create vertex buffer 创建顶点缓冲区
var vertexDataBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vertexDataBuffer);
gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.STATIC_DRAW);

// Layout of our data in the vertex buffer 顶点缓冲区中数据的布局
var positionHandle = getAttribLocation(program, "position");

gl.enableVertexAttribArray(positionHandle);
// 配置顶点属性指针
gl.vertexAttribPointer(
	// 顶点着色器中位置属性变量的句柄（handle），用于指定顶点数据中的位置信息
	positionHandle,
	// 每个位置属性变量由两个值组成，即vec2类型
	2,
	// 每个位置属性变量的组成元素是浮点数类型
	gl.FLOAT, 
	// 不需要对属性值进行归一化处理
	false, 
	// 每个顶点的位置属性在顶点数据缓冲区中占用的字节数。两个浮点数（每个浮点数占用4个字节）来表示每个顶点的位置属性
	2 * 4,
	// 从顶点数据缓冲区的起始位置开始读取顶点数据
	0 
);

//Set uniform handle 设置统一变量
var timeHandle = getUniformLocation(program, "time");
var widthHandle = getUniformLocation(program, "width");
var heightHandle = getUniformLocation(program, "height");

gl.uniform1f(widthHandle, window.innerWidth);
gl.uniform1f(heightHandle, window.innerHeight);

var lastFrame = Date.now();
var thisFrame;

function draw() {
	//Update time
	thisFrame = Date.now();
	time += (thisFrame - lastFrame) / 1000;
	lastFrame = thisFrame;

	//Send uniforms to program
	gl.uniform1f(timeHandle, time);
	//Draw a triangle strip connecting vertices 0-4 绘制一个连接顶点0-4的三角形带
	gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

	// Window 执行一个动画，并且要求浏览器在下次重绘之前调用指定的回调函数更新动画
	requestAnimationFrame(draw);
}

draw();
