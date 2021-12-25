package com.asav.android;

import com.asav.android.db.ClassifierResult;
import com.asav.android.db.ImageClassificationData;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;


/**
 * Created by avsavchenko.
 */
public class EmotionData implements ClassifierResult,Serializable {
    public float[] emotionScores=null;

    public ImageClassificationData scenes;
    public ImageClassificationData events;

    public static final float SCENE_DISPLAY_THRESHOLD = 0.1f; //0.3f;
    public static final float SCENE_CATEGORY_THRESHOLD = 0.2f; //0.3f;

    public static final float EVENT_DISPLAY_THRESHOLD = -0.3f;
    public static final float EVENT_CATEGORY_THRESHOLD = 0.1f;

    public EmotionData(){

    }
//    public EmotionData(float[] emotionScores){
//        this.emotionScores = new float[emotionScores.length];
//        System.arraycopy(emotionScores, 0, this.emotionScores, 0, emotionScores.length);
//    }
    public EmotionData(TreeMap<String,Integer> sceneLabels2Index, TreeMap<String,Float> scene2Score,
                     TreeMap<String,Integer> eventLabels2Index, TreeMap<String,Float> event2Score, float[] emotionScores){
        this.scenes =new ImageClassificationData(sceneLabels2Index,scene2Score,SCENE_DISPLAY_THRESHOLD);
        this.events =new ImageClassificationData(eventLabels2Index,event2Score,EVENT_DISPLAY_THRESHOLD);

        this.emotionScores = new float[emotionScores.length];
        System.arraycopy(emotionScores, 0, this.emotionScores, 0, emotionScores.length);
    }

    private static String[] emotions={"","Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"};
    public static String getEmotion(float[] emotionScores){
        int bestInd=-1;
        if (emotionScores!=null){
            float maxScore=0;
            for(int i=0;i<emotionScores.length;++i){
                if(maxScore<emotionScores[i]){
                    maxScore=emotionScores[i];
                    bestInd=i;
                }
            }
        }
        return emotions[bestInd+1];
    }
    public List<String> getMostReliableCategories() {
        List<String> res=new ArrayList<>();
        scenes.getMostReliableCategories(res,SCENE_CATEGORY_THRESHOLD);
        events.getMostReliableCategories(res,EVENT_CATEGORY_THRESHOLD);
        return res;
    }
    public String toString(){
        return getEmotion(emotionScores);
    }

}
