package backend.backend.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "predictions")
public class Prediction {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false)
    private String modelName;
    
    @Column(nullable = false)
    private LocalDateTime predictionDate;
    
    @Column(nullable = false)
    private String predictedDisease;
    
    @Column(nullable = false)
    private Double confidenceScore;
    
    @Column(nullable = false)
    private Boolean isConfirmedByDoctor = false;
    
    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "medical_record_id", nullable = false)
    private MedicalRecord medicalRecord;
    
    @OneToOne(mappedBy = "prediction", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private Review review;
    
    @PrePersist
    protected void onCreate() {
        predictionDate = LocalDateTime.now();
        if (modelName == null) {
            modelName = "Diabetes Neural Network v1.0";
        }
    }
    
    // Constructors
    public Prediction() {}
    
    public Prediction(MedicalRecord medicalRecord, String predictedDisease, 
                     Double confidenceScore) {
        this.medicalRecord = medicalRecord;
        this.predictedDisease = predictedDisease;
        this.confidenceScore = confidenceScore;
        this.modelName = "Diabetes Neural Network v1.0";
    }
    
    // Getters and Setters
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
    }
    
    public String getModelName() {
        return modelName;
    }
    
    public void setModelName(String modelName) {
        this.modelName = modelName;
    }
    
    public LocalDateTime getPredictionDate() {
        return predictionDate;
    }
    
    public void setPredictionDate(LocalDateTime predictionDate) {
        this.predictionDate = predictionDate;
    }
    
    public String getPredictedDisease() {
        return predictedDisease;
    }
    
    public void setPredictedDisease(String predictedDisease) {
        this.predictedDisease = predictedDisease;
    }
    
    public Double getConfidenceScore() {
        return confidenceScore;
    }
    
    public void setConfidenceScore(Double confidenceScore) {
        this.confidenceScore = confidenceScore;
    }
    
    public Boolean getIsConfirmedByDoctor() {
        return isConfirmedByDoctor;
    }
    
    public void setIsConfirmedByDoctor(Boolean isConfirmedByDoctor) {
        this.isConfirmedByDoctor = isConfirmedByDoctor;
    }
    
    public MedicalRecord getMedicalRecord() {
        return medicalRecord;
    }
    
    public void setMedicalRecord(MedicalRecord medicalRecord) {
        this.medicalRecord = medicalRecord;
    }
    
    public Review getReview() {
        return review;
    }
    
    public void setReview(Review review) {
        this.review = review;
    }
    
    public String getConfidencePercentage() {
        return String.format("%.1f%%", confidenceScore * 100);
    }
} 