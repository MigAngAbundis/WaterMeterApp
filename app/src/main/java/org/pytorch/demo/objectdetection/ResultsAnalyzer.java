package org.pytorch.demo.objectdetection;

import java.util.ArrayList;
import java.util.Collections;

public class ResultsAnalyzer {

    public static Vector2 computeRegressionLine(ArrayList<Result> results) {
        ArrayList<Vector2> centers = computeLineCenters(results);
        int n = centers.size();
        float sumX = 0;
        float sumY = 0;
        float sumXX = 0;
        float sumXY = 0;

        for (Vector2 point : centers) {
            float xi = point.x;
            float yi = point.y;
            sumX += xi;
            sumY += yi;
            sumXX += xi * xi;
            sumXY += xi * yi;
        }

        float m = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        float b = (sumY - m * sumX) / n;

        return new Vector2(m,b);
    }

    private static ArrayList<Vector2> computeLineCenters(ArrayList<Result> results) {
        ArrayList<Vector2> centers = new ArrayList<>();
        for(Result r : results) {
            float w = r.rect.right-r.rect.left;
            float h = r.rect.bottom - r.rect.top;
            float x = r.rect.left + (w/2);
            float y = r.rect.top + (h/2);
            centers.add(new Vector2(x,y));
        }
        return centers;
    }

    private static ArrayList<ResultWithCenter> computeCentersWithResult(ArrayList<Result> results) {
        ArrayList<Vector2> centers = computeLineCenters(results);
        ArrayList<ResultWithCenter> rwc = new ArrayList<>();
        for(int i = 0; i < results.size(); i++) {
            rwc.add(new ResultWithCenter(results.get(i),centers.get(i)));
        }
        return rwc;
    }

    public static String computeReading(ArrayList<Result> results) {
        StringBuilder sb = new StringBuilder();
        ArrayList<ResultWithCenter> centers = computeCentersWithResult(results);
        Collections.sort(centers);
        for(ResultWithCenter r : centers) {
            sb.append(r.r.classIndex);
        }
        return sb.toString();
    }

    private static class ResultWithCenter implements Comparable<ResultWithCenter>{
        Result r;
        Vector2 center;
        ResultWithCenter(Result r, Vector2 center) {
            this.r = r;
            this.center = center;
        }

        @Override
        public int compareTo(ResultWithCenter resultWithCenter) {
            float d = this.center.x - resultWithCenter.center.x;
            if(d < 0) return -1;
            if(d > 0) return 1;
            return 0;
        }
    }

    public static class Vector2 {
        float x,y;
        Vector2(float x, float y) {this.x = x; this.y = y;}
    }

}
