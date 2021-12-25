package com.asav.android;

import android.content.Context;
import android.util.Log;

import com.asav.android.db.ClassifierResult;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Created by avsavchenko.
 */
public class EmotionTfLiteClassifier extends TfLiteClassifier{

    /** Tag for the {@link Log}. */
    private static final String TAG = "EmotionTfLite";

    private static final String MODEL_FILE = "emotions_mobilenet_7.tflite";

    private static final String SCENES_LABELS_FILE =
            "scenes_places_emo.txt";
    private static final String EVENTS_LABELS_FILE =
            "events.txt";

    public TreeMap<String,Integer> sceneLabels2Index =new TreeMap<>();
    private ArrayList<String> sceneLabels = new ArrayList<String>();
    private Map<String,Integer> labels2HighLevelCategories = new HashMap<>();
    private Set<Integer> filteredIndices=new HashSet<>();

    public TreeMap<String,Integer> eventLabels2Index =new TreeMap<>();
    private ArrayList<String> eventLabels = new ArrayList<String>();

    public EmotionTfLiteClassifier(final Context context) throws IOException {
        super(context,MODEL_FILE);

        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(context.getAssets().open(EVENTS_LABELS_FILE)));
            String line;
            int line_ind=0;
            while ((line = br.readLine()) != null) {
                ++line_ind;
                line=line.split("#")[0].trim();
                String[] categoryInfo=line.split("=");
                String category=categoryInfo[0];
                eventLabels.add(category);

                int highLevelCategory=Integer.parseInt(categoryInfo[1]);
                labels2HighLevelCategories.put(category,highLevelCategory);
            }
            br.close();

            TreeSet<String> labelsSorted=new TreeSet<>();
            for (int i = 0; i < eventLabels.size(); ++i) {
                String event = eventLabels.get(i);
                if(!labelsSorted.contains(event))
                    labelsSorted.add(event);
            }

            int index=0;
            for(String label : labelsSorted) {
                eventLabels2Index.put(label, index);
                ++index;
            }
        } catch (IOException e) {
            throw new RuntimeException("Problem reading event label file!" , e);
        }

        try {
            br = new BufferedReader(new InputStreamReader(context.getAssets().open(SCENES_LABELS_FILE)));
            String line;
            int line_ind=0;
            while ((line = br.readLine()) != null) {
                ++line_ind;
                line=line.split("#")[0].trim();
                String[] categoryInfo=line.split("=");
                String category=categoryInfo[0];
                sceneLabels.add(category);

                int highLevelCategory=Integer.parseInt(categoryInfo[1]);
                labels2HighLevelCategories.put(category,highLevelCategory);
            }
            br.close();

            TreeSet<String> labelsSorted=new TreeSet<>();
            for (int i = 0; i < sceneLabels.size(); ++i) {
                if(filteredIndices.contains(i))
                    continue;
                String scene= sceneLabels.get(i);
                if(!labelsSorted.contains(scene))
                    labelsSorted.add(scene);
            }

            int index=0;
            for(String label : labelsSorted) {
                sceneLabels2Index.put(label, index);
                ++index;
            }
        } catch (IOException e) {
            throw new RuntimeException("Problem reading scene label file!" , e);
        }
    }

    protected void addPixelValue(int val) {
        imgData.putFloat((val & 0xFF) - 103.939f);
        imgData.putFloat(((val >> 8) & 0xFF) - 116.779f);
        imgData.putFloat(((val >> 16) & 0xFF) - 123.68f);
    }

    protected ClassifierResult getResults(float[][][] outputs) {
        final float[] emotions_scores = outputs[0][0];

        TreeMap<String,Float> scene2Score=new TreeMap<>();

        scene2Score.put("Anger", (float)0);
        scene2Score.put("Disgust", (float)0);
        scene2Score.put("Fear", (float)0);
        scene2Score.put("Happiness", (float)0);
        scene2Score.put("Neutral", (float)0);
        scene2Score.put("Sadness", (float)0);
        scene2Score.put("Surprise", (float)0);

        int max_ind = 0;
        float max = 0;
        for (int i = 0; i < 7; ++i) {
            if (max < emotions_scores[i]) {
                max = emotions_scores[i];
                max_ind = i;
            }
        }
        String[] emotions={"Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"};
        String emo = emotions[max_ind];
        scene2Score.put(emo, max);

        TreeMap<String,Float> event2Score=new TreeMap<>();

        EmotionData res=new EmotionData(sceneLabels2Index,scene2Score,eventLabels2Index,event2Score, emotions_scores);
        return res;
    }

    public int getHighLevelCategory(String category){
        int res=-1;
        if(labels2HighLevelCategories.containsKey(category))
            res= labels2HighLevelCategories.get(category);
        return res;
    }
}
