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

import com.asav.android.mtcnn.Box;
import com.asav.android.mtcnn.MTCNNModel;

/**
 * Created by avsavchenko.
 */
public abstract class TfLiteClassifier {

    /** Tag for the {@link Log}. */
    private static final String TAG = "TfLiteClassifier";

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    protected Interpreter tflite_emo;

    /* Preallocated buffers for storing image data in. */
    private int[] intValues = null;
    protected ByteBuffer imgData = null;
    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    private int imageSizeX=224,imageSizeY=224;
    private float[][][] outputs;
    Map<Integer, Object> outputMap = new HashMap<>();
    private MTCNNModel mtcnnFaceDetector=null;
    private static int minFaceSize=40;

    private int imageSizeX_emo=224,imageSizeY_emo=224;
    protected ByteBuffer imgData_emo = null;
    private float[][][] outputs_emo;
    Map<Integer, Object> outputMap_emo = new HashMap<>();

    public TfLiteClassifier(final Context context, String model_path, String model_path_emo) throws IOException {

        try {
            mtcnnFaceDetector =MTCNNModel.Companion.create(context.getAssets());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing MTCNNModel!"+e);
        }

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

        Interpreter.Options options_emo = (new Interpreter.Options()).setNumThreads(4);//.addDelegate(delegate);
        if (hasGPU) {
            org.tensorflow.lite.gpu.GpuDelegate.Options opt=new org.tensorflow.lite.gpu.GpuDelegate.Options();
            opt.setInferencePreference(org.tensorflow.lite.gpu.GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
            org.tensorflow.lite.gpu.GpuDelegate delegate = new org.tensorflow.lite.gpu.GpuDelegate(opt);
            options_emo.addDelegate(delegate);
        }

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

        MappedByteBuffer tfliteModel_emo= FileUtil.loadMappedFile(context,model_path_emo);
        tflite_emo = new Interpreter(tfliteModel_emo,options_emo);
        tflite_emo.allocateTensors();
        inputShape=tflite_emo.getInputTensor(0).shape();
        imageSizeX_emo=inputShape[1];
        imageSizeY_emo=inputShape[2];
        intValues = new int[imageSizeX_emo * imageSizeY_emo];
        imgData_emo =ByteBuffer.allocateDirect(imageSizeX_emo*imageSizeY_emo* inputShape[3]*getNumBytesPerChannel());
        imgData_emo.order(ByteOrder.nativeOrder());

        outputCount=tflite_emo.getOutputTensorCount();
        outputs_emo=new float[outputCount][1][];
        for(int i = 0; i< outputCount; ++i) {
            int[] shape=tflite_emo.getOutputTensor(i).shape();
            int numOFFeatures = shape[1];
            Log.i(TAG, "Read output layer size is " + numOFFeatures);
            outputs_emo[i][0] = new float[numOFFeatures];
            ByteBuffer ith_output = ByteBuffer.allocateDirect( numOFFeatures* getNumBytesPerChannel());  // Float tensor, shape 3x2x4
            ith_output.order(ByteOrder.nativeOrder());
            outputMap_emo.put(i, ith_output);
        }
    }
    protected abstract void addPixelValue(int val);
    protected abstract void addPixelValue_emo(int val);


    /** Classifies a frame from the preview stream. */
    public ClassifierResult classifyFrame(Bitmap bmp) {
        Object[] inputs={null};

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

        for (Box box : bboxes) {

            android.graphics.Rect bbox = new android.graphics.Rect(Math.max(0, bmp.getWidth() * box.left() / resizedBitmap.getWidth()),
                    Math.max(0, bmp.getHeight() * box.top() / resizedBitmap.getHeight()),
                    bmp.getWidth() * box.right() / resizedBitmap.getWidth(),
                    bmp.getHeight() * box.bottom() / resizedBitmap.getHeight()
            );

            Bitmap faceBitmap = Bitmap.createBitmap(bmp, bbox.left, bbox.top, bbox.width(), bbox.height());
            bmp = Bitmap.createScaledBitmap(faceBitmap, imageSizeX, imageSizeY, false);

            break; // Будем обрабатывать 1 лицо на фото.
        }

        bmp.getPixels(intValues, 0, bmp.getWidth(), 0, 0, bmp.getWidth(), bmp.getHeight());
        if (imgData == null) {
            return null;
        }
        imgData.rewind();
        imgData_emo.rewind();

        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < imageSizeX; ++i) {
            for (int j = 0; j < imageSizeY; ++j) {
                final int val = intValues[pixel++];
                addPixelValue(val);
                addPixelValue_emo(val);
            }
        }
        inputs[0] = imgData;
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

        try {
            inputs[0] = imgData_emo;
            tflite_emo.runForMultipleInputsOutputs(inputs, outputMap_emo);
        } catch (Exception e) {
            Log.e(TAG, "While get emo exception thrown: ", e);
        }

        return getResults(outputs, outputs_emo);
    }

    public void close() {
        tflite.close();
    }

    protected abstract ClassifierResult getResults(float[][][] outputs, float[][][] outputs_emo);
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
