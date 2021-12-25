package com.asav.android.db;

import java.io.Serializable;

/**
 * Created by avsavchenko.
 */
public class ImageAnalysisResults implements Serializable {
    public String filename=null;
    public ClassifierResult scene=null;
    public EXIFData locations=null;

    public ImageAnalysisResults() {}

    public ImageAnalysisResults(String filename, ClassifierResult scene, EXIFData locations){
        this.filename=filename;
        this.scene=scene;
        this.locations=locations;
    }
    public ImageAnalysisResults(ClassifierResult scene){
        this.scene=scene;
    }
}
