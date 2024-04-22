package org.pytorch.demo.objectdetection;

import android.content.Intent;
import android.net.Uri;

import androidx.documentfile.provider.DocumentFile;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.logging.Logger;

public class ValidationAnalyst {

    Logger log;
    DocumentFile imagesFolder;
    DocumentFile labelsFolder;

    public ValidationAnalyst(DocumentFile df) {
        log = Logger.getLogger("ValAnalyst");
        imagesFolder = df.findFile("images");
        labelsFolder = df.findFile("labels");
        if(imagesFolder != null && labelsFolder != null) {
            log.info("Images: "+imagesFolder.listFiles().length);
            log.info("Labels: "+labelsFolder.listFiles().length);
            buildDataset(imagesFolder,labelsFolder);
        }else {
            log.info("Error getting images and labels folder.");
        }
    }

    private ArrayList<DatasetEntry> buildDataset(DocumentFile imagesFolder, DocumentFile labelsFolder) {
        DocumentFile[] images = imagesFolder.listFiles();
        DocumentFile[] labels = labelsFolder.listFiles();
        int c = 0;

        ArrayList<DatasetEntry> dataset = new ArrayList<>();

        // Create a mapping between image names and label names
        Map<String, DocumentFile> labelMap = new HashMap<>();
        for (DocumentFile labelf : labels) {
            String n = labelf.getName();
            labelMap.put(n.substring(0, n.length() - 4), labelf);
        }

        for (DocumentFile imfile : images) {
            String name = imfile.getName();
            DocumentFile labelfile = labelMap.get(name.substring(0, name.length() - 4));

            if (labelfile == null) {
                log.info("Found img without label: " + name);
            } else {
                dataset.add(new DatasetEntry(imfile,labelfile));
                c++;
            }
        }

        log.info("Built dataset for " + c + " images.");
        return dataset;
    }

    public static class DatasetEntry {
        DocumentFile imageFile;
        DocumentFile labelFile;

        public DatasetEntry(DocumentFile a, DocumentFile b) {
            imageFile = a;
            labelFile = b;
        }

    }

}
