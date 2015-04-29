// Copyright 2011 <chaishushan@gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package opencv

//#include "opencv.h"
//#cgo linux  pkg-config: opencv
//#cgo darwin pkg-config: opencv
//#cgo windows LDFLAGS: -lopencv_core242.dll -lopencv_imgproc242.dll -lopencv_photo242.dll -lopencv_highgui242.dll -lstdc++
import "C"
import (
	"fmt"
	"math"
	"unsafe"
)

func init() {
}

/****************************************************************************************\
* Array allocation, deallocation, initialization and access to elements       *
\****************************************************************************************/

func Alloc(size int) unsafe.Pointer {
	return unsafe.Pointer(C.cvAlloc(C.size_t(size)))
}
func Free(p unsafe.Pointer) {
	C.cvFree_(p)
}

/* Allocates and initializes IplImage header */
func CreateImageHeader(w, h, depth, channels int) *IplImage {
	hdr := C.cvCreateImageHeader(
		C.cvSize(C.int(w), C.int(h)),
		C.int(depth),
		C.int(channels),
	)
	return (*IplImage)(hdr)
}

/* Inializes IplImage header */
func (img *IplImage) InitHeader(w, h, depth, channels, origin, align int) {
	C.cvInitImageHeader(
		(*C.IplImage)(img),
		C.cvSize(C.int(w), C.int(h)),
		C.int(depth),
		C.int(channels),
		C.int(origin),
		C.int(align),
	)
}

/* Creates IPL image (header and data) */
func CreateImage(w, h, depth, channels int) *IplImage {
	size := C.cvSize(C.int(w), C.int(h))
	img := C.cvCreateImage(size, C.int(depth), C.int(channels))
	return (*IplImage)(img)
}

/* Releases (i.e. deallocates) IPL image header */
func (img *IplImage) ReleaseHeader() {
	img_c := (*C.IplImage)(img)
	C.cvReleaseImageHeader(&img_c)
}

/* Releases IPL image header and data */
func (img *IplImage) Release() {
	img_c := (*C.IplImage)(img)
	C.cvReleaseImage(&img_c)
}

/* Creates a copy of IPL image (widthStep may differ) */
func (img *IplImage) Clone() *IplImage {
	p := C.cvCloneImage((*C.IplImage)(img))
	return (*IplImage)(p)
}

/* Sets a Channel Of Interest (only a few functions support COI) -
   use cvCopy to extract the selected channel and/or put it back */
func (img *IplImage) SetCOI(coi int) {
	C.cvSetImageCOI((*C.IplImage)(img), C.int(coi))
}

/* Retrieves image Channel Of Interest */
func (img *IplImage) GetCOI() int {
	coi := C.cvGetImageCOI((*C.IplImage)(img))
	return int(coi)
}

/* Sets image ROI (region of interest) (COI is not changed) */
func (img *IplImage) SetROI(rect Rect) {
	C.cvSetImageROI((*C.IplImage)(img), C.CvRect(rect))
}

/* Resets image ROI and COI */
func (img *IplImage) ResetROI() {
	C.cvResetImageROI((*C.IplImage)(img))
}

/* Retrieves image ROI */
func (img *IplImage) GetROI() Rect {
	r := C.cvGetImageROI((*C.IplImage)(img))
	return Rect(r)
}

// mat step
const (
	CV_AUTOSTEP = C.CV_AUTOSTEP
)

/* Allocates and initalizes CvMat header */
func CreateMatHeader(rows, cols, type_ int) *Mat {
	mat := C.cvCreateMatHeader(
		C.int(rows), C.int(cols), C.int(type_),
	)
	return (*Mat)(mat)
}

/* Allocates and initializes CvMat header and allocates data */
func CreateMat(rows, cols, type_ int) *Mat {
	mat := C.cvCreateMat(
		C.int(rows), C.int(cols), C.int(type_),
	)
	return (*Mat)(mat)
}

/* Initializes CvMat header */
func (mat *Mat) InitHeader(rows, cols, type_ int, data unsafe.Pointer, step int) {
	C.cvInitMatHeader(
		(*C.CvMat)(mat),
		C.int(rows),
		C.int(cols),
		C.int(type_),
		data,
		C.int(step),
	)
}

/* Releases CvMat header and deallocates matrix data
   (reference counting is used for data) */
func (mat *Mat) Release() {
	mat_c := (*C.CvMat)(mat)
	C.cvReleaseMat(&mat_c)
}

/* Decrements CvMat data reference counter and deallocates the data if
   it reaches 0 */
func DecRefData(arr Arr) {
	C.cvDecRefData(unsafe.Pointer(arr))
}

/* Increments CvMat data reference counter */
func IncRefData(arr Arr) {
	C.cvIncRefData(unsafe.Pointer(arr))
}

/* Creates an exact copy of the input matrix (except, may be, step value) */
func (mat *Mat) Clone() *Mat {
	mat_new := C.cvCloneMat((*C.CvMat)(mat))
	return (*Mat)(mat_new)
}

/* Makes a new matrix from <rect> subrectangle of input array.
   No data is copied */
func GetSubRect(arr Arr, submat *Mat, rect Rect) *Mat {
	mat_new := C.cvGetSubRect(
		unsafe.Pointer(arr),
		(*C.CvMat)(submat),
		(C.CvRect)(rect),
	)
	return (*Mat)(mat_new)
}

//#define cvGetSubArr cvGetSubRect

/* Selects row span of the input array: arr(start_row:delta_row:end_row,:)
   (end_row is not included into the span). */
func GetRows(arr Arr, submat *Mat, start_row, end_row, delta_row int) *Mat {
	mat_new := C.cvGetRows(
		unsafe.Pointer(arr),
		(*C.CvMat)(submat),
		C.int(start_row),
		C.int(end_row),
		C.int(delta_row),
	)
	return (*Mat)(mat_new)
}
func GetRow(arr Arr, submat *Mat, row int) *Mat {
	mat_new := C.cvGetRow(
		unsafe.Pointer(arr),
		(*C.CvMat)(submat),
		C.int(row),
	)
	return (*Mat)(mat_new)
}

/* Selects column span of the input array: arr(:,start_col:end_col)
   (end_col is not included into the span) */
func GetCols(arr Arr, submat *Mat, start_col, end_col int) *Mat {
	mat_new := C.cvGetCols(
		unsafe.Pointer(arr),
		(*C.CvMat)(submat),
		C.int(start_col),
		C.int(end_col),
	)
	return (*Mat)(mat_new)
}
func GetCol(arr Arr, submat *Mat, col int) *Mat {
	mat_new := C.cvGetCol(
		unsafe.Pointer(arr),
		(*C.CvMat)(submat),
		C.int(col),
	)
	return (*Mat)(mat_new)
}

/* Select a diagonal of the input array.
   (diag = 0 means the main diagonal, >0 means a diagonal above the main one,
   <0 - below the main one).
   The diagonal will be represented as a column (nx1 matrix). */
func GetDiag(arr Arr, submat *Mat, diag int) *Mat {
	mat_new := C.cvGetDiag(
		unsafe.Pointer(arr),
		(*C.CvMat)(submat),
		C.int(diag),
	)
	return (*Mat)(mat_new)
}

/* low-level scalar <-> raw data conversion functions */
func ScalarToRawData(scalar *Scalar, data unsafe.Pointer, type_, extend_to_12 int) {
	C.cvScalarToRawData(
		(*C.CvScalar)(scalar),
		data,
		C.int(type_),
		C.int(extend_to_12),
	)
}
func RawDataToScalar(data unsafe.Pointer, type_ int, scalar *Scalar) {
	C.cvRawDataToScalar(
		data,
		C.int(type_),
		(*C.CvScalar)(scalar),
	)
}

/* Allocates and initializes CvMatND header */
func CreateMatNDHeader(sizes []int, type_ int) *MatND {
	dims := C.int(len(sizes))
	sizes_c := make([]C.int, len(sizes))
	for i := 0; i < len(sizes); i++ {
		sizes_c[i] = C.int(sizes[i])
	}

	mat := C.cvCreateMatNDHeader(
		dims, (*C.int)(&sizes_c[0]), C.int(type_),
	)
	return (*MatND)(mat)
}

/* Allocates and initializes CvMatND header and allocates data */
func CreateMatND(sizes []int, type_ int) *MatND {
	dims := C.int(len(sizes))
	sizes_c := make([]C.int, len(sizes))
	for i := 0; i < len(sizes); i++ {
		sizes_c[i] = C.int(sizes[i])
	}

	mat := C.cvCreateMatND(
		dims, (*C.int)(&sizes_c[0]), C.int(type_),
	)
	return (*MatND)(mat)
}

/* Initializes preallocated CvMatND header */
func (mat *MatND) InitMatNDHeader(sizes []int, type_ int, data unsafe.Pointer) {
	dims := C.int(len(sizes))
	sizes_c := make([]C.int, len(sizes))
	for i := 0; i < len(sizes); i++ {
		sizes_c[i] = C.int(sizes[i])
	}

	C.cvInitMatNDHeader(
		(*C.CvMatND)(mat),
		dims, (*C.int)(&sizes_c[0]), C.int(type_),
		data,
	)
}

/* Releases CvMatND */
func (mat *MatND) Release() {
	mat_c := (*C.CvMatND)(mat)
	C.cvReleaseMatND(&mat_c)
}

/* Creates a copy of CvMatND (except, may be, steps) */
func (mat *MatND) Clone() *MatND {
	mat_c := (*C.CvMatND)(mat)
	mat_ret := C.cvCloneMatND(mat_c)
	return (*MatND)(mat_ret)
}

/* Allocates and initializes CvSparseMat header and allocates data */
func CreateSparseMat(sizes []int, type_ int) *SparseMat {
	dims := C.int(len(sizes))
	sizes_c := make([]C.int, len(sizes))
	for i := 0; i < len(sizes); i++ {
		sizes_c[i] = C.int(sizes[i])
	}

	mat := C.cvCreateSparseMat(
		dims, (*C.int)(&sizes_c[0]), C.int(type_),
	)
	return (*SparseMat)(mat)
}

/* Releases CvSparseMat */
func (mat *SparseMat) Release() {
	mat_c := (*C.CvSparseMat)(mat)
	C.cvReleaseSparseMat(&mat_c)
}

/* Creates a copy of CvSparseMat (except, may be, zero items) */
func (mat *SparseMat) Clone() *SparseMat {
	mat_c := (*C.CvSparseMat)(mat)
	mat_ret := C.cvCloneSparseMat(mat_c)
	return (*SparseMat)(mat_ret)
}

/* Initializes sparse array iterator
   (returns the first node or NULL if the array is empty) */
func (mat *SparseMat) InitSparseMatIterator(iter *SparseMatIterator) *SparseNode {
	mat_c := (*C.CvSparseMat)(mat)
	node := C.cvInitSparseMatIterator(mat_c, (*C.CvSparseMatIterator)(iter))
	return (*SparseNode)(node)
}

// returns next sparse array node (or NULL if there is no more nodes)
func (iter *SparseMatIterator) Next() *SparseNode {
	node := C.cvGetNextSparseNode((*C.CvSparseMatIterator)(iter))
	return (*SparseNode)(node)
}

/******** matrix iterator: used for n-ary operations on dense arrays *********/

// P290

/* Returns width and height of array in elements */
func GetSizeWidth(img *IplImage) int {
	size := C.cvGetSize(unsafe.Pointer(img))
	w := int(size.width)
	return w
}
func GetSizeHeight(img *IplImage) int {
	size := C.cvGetSize(unsafe.Pointer(img))
	w := int(size.height)
	return w
}
func GetSize(img *IplImage) Size {
	sz := C.cvGetSize(unsafe.Pointer(img))
	return Size{int(sz.width), int(sz.height)}

}

/* Copies source array to destination array */
func Copy(src, dst, mask *IplImage) {
	C.cvCopy(unsafe.Pointer(src), unsafe.Pointer(dst), unsafe.Pointer(mask))
}

//CVAPI(void)  cvCopy( const CvArr* src, CvArr* dst,
//                     const CvArr* mask CV_DEFAULT(NULL) );

/* Clears all the array elements (sets them to 0) */
func Zero(img *IplImage) {
	C.cvSetZero(unsafe.Pointer(img))
}

//CVAPI(void)  cvSetZero( CvArr* arr );
//#define cvZero  cvSetZero

/****************************************************************************************\
*                   Arithmetic, logic and comparison operations               *
\****************************************************************************************/

/* dst(idx) = ~src(idx) */
func Not(src, dst *IplImage) {
	C.cvNot(unsafe.Pointer(src), unsafe.Pointer(dst))
}

//CVAPI(void) cvNot( const CvArr* src, CvArr* dst );

func And(src, src2, dst, mask *IplImage) {
	C.cvAnd(unsafe.Pointer(src), unsafe.Pointer(src2), unsafe.Pointer(dst), unsafe.Pointer(mask))
}

//CVAPI(void) cvAnd( const CvArr* src, const CvArr* src, CvArr* dst, CvArr* mask);

/****************************************************************************************\
*                                Math operations                              *
\****************************************************************************************/

func Add(src, src2, dst, mask *IplImage) {
	C.cvAdd(unsafe.Pointer(src), unsafe.Pointer(src2), unsafe.Pointer(dst), unsafe.Pointer(mask))
}

//CVAPI(void) cvAdd( const CvArr* src, const CvArr* src2, CvArr* dst, CvArr* mask);

func Sub(src, src2, dst, mask *IplImage) {
	C.cvSub(unsafe.Pointer(src), unsafe.Pointer(src2), unsafe.Pointer(dst), unsafe.Pointer(mask))
}

//CVAPI(void)  cvSub( const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask CV_DEFAULT(NULL));

func Divide(src, src2, dst *IplImage, scale float64) {
	C.cvDiv(unsafe.Pointer(src), unsafe.Pointer(src2), unsafe.Pointer(dst), C.double(scale))
}

//CVAPI(void) cvDiv( const CvArpr* src, const CvArr* src2, CvArr* dst, scale double);

func ConvertScale(src, dest *IplImage, scale, shift float64) {
	C.cvConvertScale(unsafe.Pointer(src), unsafe.Pointer(dest), C.double(scale), C.double(shift))
}

/*CVAPI(void)  cvConvertScale( const CvArr* src, CvArr* dst,
double scale CV_DEFAULT(1),
double shift CV_DEFAULT(0) );*/

func Log(src, src2, dst *IplImage) {
	C.cvLog(unsafe.Pointer(src), unsafe.Pointer(dst))
}

//CVAPI(void) cvLog( const CvArr* src, CvArr* dst);

func Min(src, src2, output *IplImage) {
	C.cvMin(unsafe.Pointer(src), unsafe.Pointer(src2), unsafe.Pointer(output))
}

//CVAPI(void) cvMin( const CvArr* src1, const CvArr* src2, CvArr* dst );)

func MeanStdDev(src, mask *IplImage) (m, s [4]float64) {
	mean, stddev := Scalar{}, Scalar{}
	C.cvAvgSdv(unsafe.Pointer(src), (*C.CvScalar)(&mean), (*C.CvScalar)(&stddev), unsafe.Pointer(mask))
	for i, x := range mean.val {
		m[i] = float64(x)
	}
	for i, x := range stddev.val {
		s[i] = float64(x)
	}
	return m, s
}

/****************************************************************************************\
*                                Matrix operations                            *
\****************************************************************************************/

/****************************************************************************************\
*                                    Array Statistics                         *
\****************************************************************************************/

/****************************************************************************************\
*                      Discrete Linear Transforms and Related Functions       *
\****************************************************************************************/

/****************************************************************************************\
*                              Dynamic data structures                        *
\****************************************************************************************/

/****************************************************************************************\
*                                            Contours                          *
\****************************************************************************************/
// Contours wraps the memory allocation of CvSeq and the pointer
// length used for storing and walking through arrays of CvContour
type Contours struct {
	mem      *C.CvMemStorage
	seq      *C.CvSeq
	elements [](*C.struct_CvSeq)
	elemSize uintptr
}

func (c *Contours) Release() {
	C.cvClearMemStorage(c.mem)
	C.cvReleaseMemStorage(&c.mem)
}

func (c *Contours) Elements() [](*C.struct_CvSeq) {
	return c.elements
}

func (c *Contours) BoundingBoxAt(index int) Rect {
	box := C.cvBoundingRect(unsafe.Pointer(c.elements[index]), 0)
	return Rect{x: box.x, y: box.y, width: box.width, height: box.height}
}

func (c *Contours) MaxBoundingBox() (Rect, int) {
	maxArea := float64(0)
	maxIndex := 0

	for i := 0; i < len(c.elements); i++ {
		area := ContourArea(*c, i)

		if area > maxArea {
			maxArea = area
			maxIndex = i
		}
	}
	box := C.cvBoundingRect(unsafe.Pointer(c.elements[maxIndex]), 0)
	return Rect{x: box.x, y: box.y, width: box.width, height: box.height}, maxIndex
}

type AreaRect struct {
	X , Y float64
	Width, Height float64
	Angle float64
}

func (c *Contours) MinAreaRect2(index int) C.struct_CvBox2D {
	box := C.cvMinAreaRect2(unsafe.Pointer(c.elements[index]), nil)
	return box
}

func CvBox2DToAreaRect(box C.struct_CvBox2D) AreaRect {
	// center:{x:403.3662 y:288.5526} size:{width:461.07642 height:606.7056} angle:-41.08366}
	return AreaRect{X: float64(box.center.x), Y: float64(box.center.y), Width: float64(box.size.width), Height: float64(box.size.height), Angle: float64(box.angle)}
}

func (c *Contours) DumpPoints() {
	tmp := (*C.struct_CvSeq)(unsafe.Pointer(c.seq))
	total := int(tmp.total)
	for i := 0; i < total; i++ {
		p := (*C.CvPoint)(unsafe.Pointer(C.cvGetSeqElem(c.seq, C.int(i))))
		fmt.Printf("point %d (x = %+v, y = %+v)\n", i, p.x, p.y)
	}
}

type Corners struct {
	TopLeft, TopRight, BottomRight, BottomLeft Point
}

func Intersections(lines []Vector, width int, height int) []Point {
	intersections := make([]Point, 0)

	for outer := 0; outer < len(lines); outer++ {
		for inner := 0; inner < len(lines); inner++ {
			p0 := lines[outer]
			x1 := float64(p0.Start.X)
			y1 := float64(p0.Start.Y)
			x2 := float64(p0.End.X)
			y2 := float64(p0.End.Y)

			p1 := lines[inner]
			y3 := float64(p1.Start.Y)
			x3 := float64(p1.Start.X)
			x4 := float64(p1.End.X)
			y4 := float64(p1.End.Y)

			d := math.Abs(((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))

			if d != 0 {
				x := ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / d
				y := ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / d
				inter := Point{X: int32(x), Y: int32(y)}
				if inter.X > 0 && inter.Y > 0 && inter.X < int32(width) && inter.Y < int32(height) {
					intersections = append(intersections, inter)
					//fmt.Printf("comparing lines %+v with %+v = %+v (Intersection at %+v)\n", p0, p1, d, inter)
				}
			}

		}
	}

	// Filter them
	maxX := Point{}
	maxY := Point{}
	minX := Point{X: 99999, Y: 99999}
	minY := Point{X: 99999, Y: 99999}

	filtered := make([]Point, 0)
	for i := 0; i < len(intersections); i++ {
		point := intersections[i]

		if !fuzzyPointIn(filtered, point, 50) {
			if point.X > maxX.X {
				maxX = point
			}

			if point.X < minX.X {
				minX = point
			}

			if point.Y > maxY.Y {
				maxY = point
			}

			if point.Y < minY.Y {
				minY = point
			}
		} else {
			fmt.Printf("point filtered out by %d pixels = %+v\n", 50, point)
		}

	}

	filtered = append(filtered, minX)
	filtered = append(filtered, minY)
	filtered = append(filtered, maxX)
	filtered = append(filtered, maxY)

	return filtered
}

func fuzzyPointIn(present []Point, search Point, fuzzy float64) bool {
	for i := 0; i < len(present); i++ {
		derX := math.Abs(float64(present[i].X - search.X))
		derY := math.Abs(float64(present[i].Y - search.Y))
		if derX < fuzzy && derY < fuzzy {
			return true
		}
	}
	return false
}

func GetCorners(box C.struct_CvBox2D) Corners {
	var points [4]C.CvPoint2D32f
	var topPoints [2]C.CvPoint2D32f
	var bottomPoints [2]C.CvPoint2D32f
	topIndex := 0
	bottomIndex := 0

	// Find minimal rotated bounding box
	ptr := unsafe.Pointer(&points)
	cCompatPointer := (*C.struct_CvPoint2D32f)(ptr)
	corners := Corners{}

	// Put an array[4] of CvPoint2D32f into points
	// referenced by ptr -> cCompatPointer
	C.cvBoxPoints(box, cCompatPointer)

	// Sort point in top and bottom parts
	for i := 0; i < 4; i++ {
		if points[i].y < box.center.y {
			topPoints[topIndex] = points[i]
			topIndex++
		} else {
			bottomPoints[bottomIndex] = points[i]
			bottomIndex++
		}
	}

	// Sort top/bottom parts into left/right and create Point structs
	if topPoints[0].x < box.center.x {
		corners.TopLeft = Point{X: int32(topPoints[0].x), Y: int32(topPoints[0].y)}
		corners.TopRight = Point{X: int32(topPoints[1].x), Y: int32(topPoints[1].y)}
	} else {
		corners.TopRight = Point{X: int32(topPoints[0].x), Y: int32(topPoints[0].y)}
		corners.TopLeft = Point{X: int32(topPoints[1].x), Y: int32(topPoints[1].y)}
	}

	if bottomPoints[0].x < box.center.x {
		corners.BottomLeft = Point{X: int32(bottomPoints[0].x), Y: int32(bottomPoints[0].y)}
		corners.BottomRight = Point{X: int32(bottomPoints[1].x), Y: int32(bottomPoints[1].y)}
	} else {
		corners.BottomLeft = Point{X: int32(bottomPoints[1].x), Y: int32(bottomPoints[1].y)}
		corners.BottomRight = Point{X: int32(bottomPoints[0].x), Y: int32(bottomPoints[0].y)}
	}

	return corners
}

func (c *Contours) Points(index int) [4]Point {
	box := C.cvMinAreaRect2(unsafe.Pointer(c.elements[index]), nil)
	var points [4]C.CvPoint2D32f
	ptr := unsafe.Pointer(&points)
	c2 := (*C.struct_CvPoint2D32f)(ptr)
	C.cvBoxPoints(box, c2)

	fmt.Printf("points = %+v\n", points)
	var thePoints [4]Point
	thePoints[0] = Point{X: int32(points[0].x), Y: int32(points[0].y)}
	thePoints[1] = Point{X: int32(points[1].x), Y: int32(points[1].y)}
	thePoints[2] = Point{X: int32(points[2].x), Y: int32(points[2].y)}
	thePoints[3] = Point{X: int32(points[3].x), Y: int32(points[3].y)}

	return thePoints
}

func (c *Contours) GetRotationMatrix2D(index int) *C.struct_CvMat {
	box := c.MinAreaRect2(index)
	mat := C.cvCreateMat(2, 3, C.CV_32FC1)
	C.cv2DRotationMatrix(box.center, C.double(box.angle), 1, mat)
	return mat
}

func GetPerspectiveTransform(srcCorners Corners, destCorners Corners) *C.struct_CvMat {
	mat := C.cvCreateMat(3, 3, C.CV_32FC1)

	var src [4]C.CvPoint2D32f
	var dest [4]C.CvPoint2D32f

	src[0].x = C.float(srcCorners.TopLeft.X)
	src[0].y = C.float(srcCorners.TopLeft.Y)
	src[1].x = C.float(srcCorners.TopRight.X)
	src[1].y = C.float(srcCorners.TopRight.Y)
	src[2].x = C.float(srcCorners.BottomRight.X)
	src[2].y = C.float(srcCorners.BottomRight.Y)
	src[3].x = C.float(srcCorners.BottomLeft.X)
	src[3].y = C.float(srcCorners.BottomLeft.Y)

	dest[0].x = C.float(destCorners.TopLeft.X)
	dest[0].y = C.float(destCorners.TopLeft.Y)
	dest[1].x = C.float(destCorners.TopRight.X)
	dest[1].y = C.float(destCorners.TopRight.Y)
	dest[2].x = C.float(destCorners.BottomRight.X)
	dest[2].y = C.float(destCorners.BottomRight.Y)
	dest[3].x = C.float(destCorners.BottomLeft.X)
	dest[3].y = C.float(destCorners.BottomLeft.Y)

	ptrSrc := unsafe.Pointer(&src)
	cSrc := (*C.struct_CvPoint2D32f)(ptrSrc)

	ptrDest := unsafe.Pointer(&dest)
	cDest := (*C.struct_CvPoint2D32f)(ptrDest)

	C.cvGetPerspectiveTransform(cSrc, cDest, mat)
	return mat
}

func Rotate(src *IplImage, dest *IplImage, mat *C.struct_CvMat) {
	C.cvWarpAffine(unsafe.Pointer(src), unsafe.Pointer(dest), mat, C.CV_INTER_LINEAR|C.CV_WARP_FILL_OUTLIERS, C.cvScalarAll(128))
}

func WarpPerspective(src *IplImage, dest *IplImage, mat *C.struct_CvMat) {
	C.cvWarpPerspective(unsafe.Pointer(src), unsafe.Pointer(dest), mat, C.CV_INTER_LINEAR|C.CV_WARP_FILL_OUTLIERS, C.cvScalarAll(1))
}

// NewContours initializes an empty Contours structure
func NewContours() Contours {
	// Create contours by allocating memory through C for CvSeq
	mem := C.cvCreateMemStorage(0)
	return Contours{mem, createSeq(0, unsafe.Sizeof(C.struct_CvSeq{}), unsafe.Sizeof(C.CvPoint2DSeq{}), mem),
		[](*C.struct_CvSeq){}, unsafe.Sizeof(C.CvPoint2DSeq{})}
}

// FindContours creates a list of the contours in an 8-bit
// image. WARNING: will modify image used to find contours!
func FindContours(img *IplImage, mode int, method int, offset *Point) (Contours, int) {
	con := NewContours()
	pt := C.cvPoint(C.int(0), C.int(0))
	if offset != nil {
		pt = C.cvPoint(C.int(offset.X), C.int(offset.Y))
	}
	// Call C function which modifies CvSeq in place
	numFound := C.cvFindContours(unsafe.Pointer(img), con.mem, &con.seq,
		C.int(con.elemSize), C.int(mode), C.int(method), pt)

	cur := (*C.struct_CvSeq)(unsafe.Pointer(con.seq))
	for i := 0; i < int(numFound); i++ {
		con.elements = append(con.elements, cur)
		cur = cur.h_next
	}

	return con, int(numFound)
}

/*CVAPI(int)  cvFindContours( CvArr* image, CvMemStorage* storage, CvSeq** first_contour,
int header_size CV_DEFAULT(sizeof(CvContour)),
int mode CV_DEFAULT(CV_RETR_LIST),
int method CV_DEFAULT(CV_CHAIN_APPROX_SIMPLE),
CvPoint offset CV_DEFAULT(cvPoint(0,0)));*/

func ConvexArea(con Contours, index int) float64 {
	// Get xopointer to CvContour in CvSeq
	cur := (*C.CvSeq)(unsafe.Pointer(con.elements[index])) //getSeqElem(con.seq, index)
	// Get convex hull sequence
	seq := C.cvConvexHull2(unsafe.Pointer(cur), unsafe.Pointer(nil),
		C.CV_CLOCKWISE, C.int(0))
	// Create default slice which is whole contour
	var whole C.CvSlice
	whole.start_index = 0
	whole.end_index = CV_WHOLE_SEQ_END_INDEX
	area := C.cvContourArea(unsafe.Pointer(seq), whole, 0)
	return float64(area)
}

func ApproxPoly(con Contours) {
	src := (*C.CvSeq)(unsafe.Pointer(con.elements[0])) //getSeqElem(con.seq, index)
	mem := C.cvCreateMemStorage(0)
	seq := (*C.CvSeq)(C.cvApproxPoly(unsafe.Pointer(src), C.int(unsafe.Sizeof(C.struct_CvContour{})), (*C.struct_CvMemStorage)(mem), C.CV_POLY_APPROX_DP, 1, 1))

	tmp := (*C.struct_CvSeq)(unsafe.Pointer(seq))
	fmt.Printf("tmp = %+v\n", tmp)

	C.cvClearMemStorage(mem)
	C.cvReleaseMemStorage(&mem)
}

//CvSeq* cvApproxPoly(const void* src_seq, int header_size, CvMemStorage* storage, int method, double eps, int recursive=0 )

func ConvexityDefects(con Contours, index int) ([]float64, int) {
	// Get xopointer to CvContour in CvSeq
	cur := (*C.CvSeq)(unsafe.Pointer(con.elements[index])) //getSeqElem(con.seq, index)
	// Find convex hull of selected contour
	seq := C.cvConvexHull2(unsafe.Pointer(cur), unsafe.Pointer(nil),
		C.CV_CLOCKWISE, C.int(0))
	// Calculate defects of convex hull and contour
	seq2 := (*C.CvSeq)(C.cvConvexityDefects(unsafe.Pointer(cur), unsafe.Pointer(seq),
		(*C.CvMemStorage)(nil)))
	// Determine the total number of defects
	tmp := (*C.struct_CvSeq)(unsafe.Pointer(seq2))
	total := int(tmp.total)
	// Iterate through cvSeq and convert to Go Slice
	defects := make([]float64, total)
	for i := 0; i < total; i++ {
		ep := (*C.CvConvexityDefect)(unsafe.Pointer(C.cvGetSeqElem(seq2, C.int(i))))
		defects[i] = float64(C.float((*ep).depth))
	}
	return defects, total
}

// DrawSingleContour will draw only the indexed contour from a list of contours onto the passed in image
func DrawSingleContour(img *IplImage, con Contours, index int, extC Scalar, holeC Scalar, thickness int, lineType int, offset *Point) {
	// Set to draw single contour and get appropriately indexed
	// starting pointer
	pt := C.cvPoint(C.int(0), C.int(0))
	if offset != nil {
		pt = C.cvPoint(C.int(offset.X), C.int(offset.Y))
	}
	maxLevel := 0
	contourPtr := (*C.CvSeq)(unsafe.Pointer(con.elements[index])) //  getSeqElem(con.seq, index)))
	// Call C function which modifies image in place
	C.cvDrawContours(unsafe.Pointer(img), contourPtr, C.CvScalar(extC), C.CvScalar(holeC),
		C.int(maxLevel), C.int(thickness), C.int(lineType), pt)
}

// DrawContours will draw all contours onto passed in image
func DrawContours(img *IplImage, con Contours, extC Scalar, holeC Scalar, maxLevel int, thickness int, lineType int, offset Point) {
	// Call C function which modifies image in place
	C.cvDrawContours(unsafe.Pointer(img), con.seq, C.CvScalar(extC), C.CvScalar(holeC),
		C.int(maxLevel), C.int(thickness), C.int(lineType),
		C.cvPoint(C.int(offset.X), C.int(offset.Y)))
}

/*CVAPI(void)  cvDrawContours( CvArr *img, CvSeq* contour,
CvScalar external_color, CvScalar hole_color,
int max_level, int thickness CV_DEFAULT(1),
int line_type CV_DEFAULT(8),
CvPoint offset CV_DEFAULT(cvPoint(0,0)));*/

func ContourArea(con Contours, index int) float64 {
	// Get xopointer to CvContour in CvSeq
	cur := (*C.CvSeq)(unsafe.Pointer(con.elements[index])) //getSeqElem(con.seq, index)
	// Create default slice which is whole contour
	var whole C.CvSlice
	whole.start_index = 0
	whole.end_index = CV_WHOLE_SEQ_END_INDEX
	area := C.cvContourArea(unsafe.Pointer(cur), whole, 0)
	return float64(area)
}

/*CVAPI(double)  cvContourArea( const CvArr* contour,
CvSlice slice CV_DEFAULT(CV_WHOLE_SEQ),
int oriented CV_DEFAULT(0));*/

func ArcLength(con Contours, index int) float64 {
	// Get xopointer to CvContour in CvSeq
	cur := (*C.CvSeq)(unsafe.Pointer(con.elements[index])) //getSeqElem(con.seq, index)
	// Create default slice which is whole contour
	var whole C.CvSlice
	whole.start_index = 0
	whole.end_index = CV_WHOLE_SEQ_END_INDEX
	length := C.cvArcLength(unsafe.Pointer(cur), whole, -1)
	return float64(length)
}

type CvMoments struct {
	M00, M10, M01, M20, M11, M02, M30, M21, M12, M03 float64
	N20, N11, N02, N30, N21, N12, N03, ISM00         float64
}
type CvHuMoments struct{ H1, H2, H3, H4, H5, H6, H7 float64 }

func Moments(src *IplImage, binary int) (m CvMoments, hu CvHuMoments) {
	C.cvMoments(unsafe.Pointer(src), (*C.CvMoments)(unsafe.Pointer(&m)), C.int(binary))
	C.cvGetHuMoments((*C.CvMoments)(unsafe.Pointer(&m)), (*C.CvHuMoments)(unsafe.Pointer(&hu)))
	return m, hu
}

/****************************************************************************************\
*                                     Threshold                                 *
\****************************************************************************************/
const (
	CV_THRESH_BINARY     = int(C.CV_THRESH_BINARY)
	CV_THRESH_BINARY_INV = int(C.CV_THRESH_BINARY_INV)
	CV_THRESH_TRUNC      = int(C.CV_THRESH_TRUNC)
	CV_THRESH_TOZERO     = int(C.CV_THRESH_TOZERO)
	CV_THRESH_TOZERO_INV = int(C.CV_THRESH_TOZERO_INV)
)

// cvThreshold(const CvArr* src, CvArr* dst, double threshold, double max_value, int threshold_type)
func Threshold(src, dst *IplImage, threshold, max_value float64, threshold_type int) float64 {
	ret := C.cvThreshold(unsafe.Pointer(src), unsafe.Pointer(dst),
		C.double(threshold), C.double(max_value), C.int(threshold_type))
	return float64(ret)
}

/*func AdaptiveThreshold(src, dst cv.IplImage, max_value float64,
	method, threshold_type, block_size int param1 float64) {
	cv.AdaptiveThreshold(const CvArr* src, CvArr* dst, double max_value,
   int adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C,
   int threshold_type=CV_THRESH_BINARY,
   int block_size=3,
   double param1=5 ) */

/* cvAdaptiveThreshold(const CvArr* src, CvArr* dst, double max_value,
   int adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C,
   int threshold_type=CV_THRESH_BINARY,
   int block_size=3,
   double param1=5 ) */

/****************************************************************************************\
*                                     Hough                                 *
\****************************************************************************************/
type Circle struct {
	x, y, r float64
}

// HoughCircles finds the circles within an 8-bit, single-channel image
func HoughCircles(src *IplImage, dp float64, min_dist float64, param1 float64, param2 float64, min_radius int, max_radius int) []Circle {
	mem := C.cvCreateMemStorage(0)
	seq := (*C.CvSeq)(C.cvHoughCircles(unsafe.Pointer(src), unsafe.Pointer(mem), C.CV_HOUGH_GRADIENT, C.double(dp), C.double(min_dist), C.double(param1), C.double(param2), C.int(min_radius), C.int(max_radius)))
	tmp := (*C.struct_CvSeq)(unsafe.Pointer(seq))
	total := int(tmp.total)
	circles := make([]Circle, total)
	for i := 0; i < total; i++ {
		p := (*[3]C.float)(unsafe.Pointer(C.cvGetSeqElem(seq, C.int(i))))
		circles[i] = Circle{float64(p[0]), float64(p[1]), float64(p[2])}
	}
	C.cvClearMemStorage(mem)
	C.cvReleaseMemStorage(&mem)
	return circles
}

/* CV_IMPL CvSeq*
cvHoughCircles( CvArr* src_image, void* circle_storage,
int method, double dp, double min_dist,
double param1, double param2,
int min_radius, int max_radius ) */
type Vector struct {
	Start, End Point
}

func HoughLinesP(src *IplImage, dist float64, angle float64, threshold int64, minLineLength float64, minGap float64) []Vector {
	mem := C.cvCreateMemStorage(0)

	seq := (*C.CvSeq)(C.cvHoughLines2(unsafe.Pointer(src), unsafe.Pointer(mem), C.CV_HOUGH_PROBABILISTIC, C.double(dist), C.double(angle), C.int(threshold), C.double(minLineLength), C.double(minGap)))

	tmp := (*C.struct_CvSeq)(unsafe.Pointer(seq))
	total := int(tmp.total)

	lines := make([]Vector, total)

	for i := 0; i < total; i++ {
		p := (*[2]C.CvPoint)(unsafe.Pointer(C.cvGetSeqElem(seq, C.int(i))))
		start := Point{X: int32(p[0].x), Y: int32(p[0].y)}
		end := Point{X: int32(p[1].x), Y: int32(p[1].y)}
		lines[i] = Vector{Start: start, End: end}
	}

	C.cvClearMemStorage(mem)
	C.cvReleaseMemStorage(&mem)

	return lines
}

// CvSeq* cvHoughLines2(CvArr* image, void* line_storage, int method, double rho, double theta, int threshold, double param1=0, double param2=0 )

/****************************************************************************************\
*                                     Drawing                                 *
\****************************************************************************************/

/* Draws 4-connected, 8-connected or antialiased line segment connecting two points */
//color Scalar,
func Line(image *IplImage, pt1, pt2 Point, color Scalar, thickness, line_type, shift int) {
	C.cvLine(
		unsafe.Pointer(image),
		C.cvPoint(C.int(pt1.X), C.int(pt1.Y)),
		C.cvPoint(C.int(pt2.X), C.int(pt2.Y)),
		(C.CvScalar)(color),
		C.int(thickness), C.int(line_type), C.int(shift),
	)
	//Scalar
}

//CVAPI(void)  cvLine( CvArr* img, CvPoint pt1, CvPoint pt2,
//                     CvScalar color, int thickness CV_DEFAULT(1),
//                     int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0) );

func DrawCircle(image *IplImage, center Point, radius int, color Scalar, thickness int, lineType int, shift int) {
	C.cvCircle(
		unsafe.Pointer(image),
		C.cvPoint(C.int(center.X), C.int(center.Y)),
		C.int(radius),
		(C.CvScalar)(color),
		C.int(thickness), C.int(lineType), C.int(shift),
	)
}

// void cvCircle(CvArr* img, CvPoint center, int radius, CvScalar color, int thickness=1, int line_type=8, int shift=0 )¶
/****************************************************************************************\
*                                    System functions                         *
\****************************************************************************************/

/****************************************************************************************\
*                                    Data Persistence                         *
\****************************************************************************************/
