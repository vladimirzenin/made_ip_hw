package com.asav.android;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;

import com.asav.android.db.ClassifierResult;
import com.asav.android.mtcnn.Box;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.common.*;
import org.tensorflow.lite.support.image.*;
import org.tensorflow.lite.support.image.ops.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

/**
 * Created by avsavchenko.
 */
public abstract class TfLiteClassifier {

    /** Tag for the {@link Log}. */
    private static final String TAG = "TfLiteClassifier";

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    /* Preallocated buffers for storing image data in. */
    private int[] intValues = null;
    protected ByteBuffer imgData = null;
    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    private int imageSizeX=224,imageSizeY=224;
    private float[][][] outputs;
    Map<Integer, Object> outputMap = new HashMap<>();

    public TfLiteClassifier(final Context context, String model_path) throws IOException {
        Interpreter.Options options = (new Interpreter.Options()).setNumThreads(4);//.addDelegate(delegate);
        CompatibilityList compatList = new CompatibilityList();
        boolean hasGPU=compatList.isDelegateSupportedOnThisDevice();
        if (hasGPU) {
            org.tensorflow.lite.gpu.GpuDelegate.Options opt=new org.tensorflow.lite.gpu.GpuDelegate.Options();
            opt.setInferencePreference(org.tensorflow.lite.gpu.GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
            org.tensorflow.lite.gpu.GpuDelegate delegate = new org.tensorflow.lite.gpu.GpuDelegate(opt);
            options.addDelegate(delegate);
        }

        MappedByteBuffer tfliteModel= FileUtil.loadMappedFile(context,model_path);
        tflite = new Interpreter(tfliteModel,options);
        tflite.allocateTensors();
        int[] inputShape=tflite.getInputTensor(0).shape();
        imageSizeX=inputShape[1];
        imageSizeY=inputShape[2];
        intValues = new int[imageSizeX * imageSizeY];
        imgData =ByteBuffer.allocateDirect(imageSizeX*imageSizeY* inputShape[3]*getNumBytesPerChannel());
        imgData.order(ByteOrder.nativeOrder());

        int outputCount=tflite.getOutputTensorCount();
        outputs=new float[outputCount][1][];
        for(int i = 0; i< outputCount; ++i) {
            int[] shape=tflite.getOutputTensor(i).shape();
            int numOFFeatures = shape[1];
            Log.i(TAG, "Read output layer size is " + numOFFeatures);
            outputs[i][0] = new float[numOFFeatures];
            ByteBuffer ith_output = ByteBuffer.allocateDirect( numOFFeatures* getNumBytesPerChannel());  // Float tensor, shape 3x2x4
            ith_output.order(ByteOrder.nativeOrder());
            outputMap.put(i, ith_output);
        }
    }
    protected abstract void addPixelValue(int val);


    /** Classifies a frame from the preview stream. */
    public ClassifierResult classifyFrame(Bitmap bitmap) {
        Object[] inputs={null};



        Bitmap bmp = bitmap;

        Bitmap resizedBitmap=bmp;
        double minSize=600.0;
        double scale=Math.min(bmp.getWidth(),bmp.getHeight())/minSize;
        if(scale>1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(bmp, (int)(bmp.getWidth()/scale), (int)(bmp.getHeight()/scale), false);
            bmp=resizedBitmap;
        }
        long startTime = SystemClock.uptimeMillis();
        Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));
        Log.i(TAG, "Timecost to run mtcnn: " + Long.toString(SystemClock.uptimeMillis() - startTime));

        Bitmap tempBmp = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(tempBmp);
        Paint p = new Paint();
        p.setStyle(Paint.Style.STROKE);
        p.setAntiAlias(true);
        p.setFilterBitmap(true);
        p.setDither(true);
        p.setColor(Color.BLUE);
        p.setStrokeWidth(5);

        Paint p_text = new Paint();
        p_text.setColor(Color.WHITE);
        p_text.setStyle(Paint.Style.FILL);
        p_text.setColor(Color.GREEN);
        p_text.setTextSize(24);

        c.drawBitmap(bmp, 0, 0, null);

        //for (Box box : bboxes) {









        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        if (imgData == null) {
            return null;
        }
        imgData.rewind();
        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < imageSizeX; ++i) {
            for (int j = 0; j < imageSizeY; ++j) {
                final int val = intValues[pixel++];
                addPixelValue(val);
            }
        }
        inputs[0] = imgData;
        long startTime = SystemClock.uptimeMillis();
        tflite.runForMultipleInputsOutputs(inputs, outputMap);
        for(int i = 0; i< outputs.length; ++i) {
            ByteBuffer ith_output=(ByteBuffer)outputMap.get(i);
            ith_output.rewind();
            int len=outputs[i][0].length;
            for(int j=0;j<len;++j){
                outputs[i][0][j]=ith_output.getFloat();
            }
            ith_output.rewind();
        }
        long endTime = SystemClock.uptimeMillis();
        Log.i(TAG, "tf lite timecost to run model inference: " + Long.toString(endTime - startTime));

        return getResults(outputs);
    }

    public void close() {
        tflite.close();
    }

    protected abstract ClassifierResult getResults(float[][][] outputs);
    public int getImageSizeX() {
        return imageSizeX;
    }
    public int getImageSizeY() {
        return imageSizeY;
    }
    protected int getNumBytesPerChannel() {
        return 4; // Float.SIZE / Byte.SIZE;
    }
}
