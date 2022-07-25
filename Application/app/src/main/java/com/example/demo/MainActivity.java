package com.example.demo;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.SparseArray;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.demo.ml.AgeModel;
import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    TextView result;
    ImageView imageView;
    Button takepic, check;
    int imageSize = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.textView);
        imageView = findViewById(R.id.imageView);
        takepic = findViewById(R.id.takepic);
        check = findViewById(R.id.check);

        takepic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent image = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(image,100);
            }
        });

        check.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Bitmap img = ((BitmapDrawable)imageView.getDrawable()).getBitmap();

                FaceDetector faceDetector = new FaceDetector.Builder(getApplicationContext()).setTrackingEnabled(false).setLandmarkType(FaceDetector.ALL_LANDMARKS).setMode(FaceDetector.FAST_MODE).build();
                if(!faceDetector.isOperational()){
                    Toast.makeText(MainActivity.this,"Restart and try again!",Toast.LENGTH_SHORT).show();
                    return;
                }

                Frame frame = new Frame.Builder().setBitmap(img).build();
                SparseArray<Face> sparseArray  = faceDetector.detect(frame);


                Face face = sparseArray.valueAt(0);
                float x = face.getPosition().x;
                float y = face.getPosition().y;
                float w = face.getWidth();
                float h = face.getHeight();

                Bitmap img_crop = Bitmap.createBitmap(img, x, y, w, h);

                Bitmap img_resize = Bitmap.createScaledBitmap(img_crop, imageSize, imageSize, false);
                classifyImage(img_resize);

            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        Bitmap image = (Bitmap) data.getExtras().get("data");
        int dimension = Math.min(image.getWidth(), image.getHeight());
        image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
        imageView.setImageBitmap(image);

        super.onActivityResult(requestCode, resultCode, data);
    }

    private void classifyImage(Bitmap image) {
        try {
            AgeModel model = AgeModel.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 100, 100, 3}, DataType.FLOAT32);
            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(image);
            ByteBuffer byteBuffer = tensorImage.getBuffer();
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            AgeModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i =0; i < confidences.length; i++){
                if (confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            checkAge(maxPos);


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    private void checkAge(int maxPos) {
        if (maxPos == 0){
            result.setText("You're under 18.");
        }
        else{
            openAcitivity2();
        }
    }

    private void openAcitivity2() {
        Intent intent = new Intent(this, Activity2.class);
        startActivity(intent);
    }

}