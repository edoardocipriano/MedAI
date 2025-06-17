package backend.backend.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "reviews")
public class Review {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false)
    private LocalDateTime reviewDate;
    
    @Column(columnDefinition = "TEXT")
    private String reviewNotes;
    
    @Column(nullable = false)
    private Boolean confirmedDiagnosis;
    
    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "prediction_id", nullable = false)
    private Prediction prediction;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "doctor_id", nullable = false)
    private Doctor doctor;
    
    @PrePersist
    protected void onCreate() {
        reviewDate = LocalDateTime.now();
    }
    
    // Constructors
    public Review() {}
    
    public Review(Prediction prediction, Doctor doctor, String reviewNotes, 
                  Boolean confirmedDiagnosis) {
        this.prediction = prediction;
        this.doctor = doctor;
        this.reviewNotes = reviewNotes;
        this.confirmedDiagnosis = confirmedDiagnosis;
    }
    
    // Getters and Setters
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
    }
    
    public LocalDateTime getReviewDate() {
        return reviewDate;
    }
    
    public void setReviewDate(LocalDateTime reviewDate) {
        this.reviewDate = reviewDate;
    }
    
    public String getReviewNotes() {
        return reviewNotes;
    }
    
    public void setReviewNotes(String reviewNotes) {
        this.reviewNotes = reviewNotes;
    }
    
    public Boolean getConfirmedDiagnosis() {
        return confirmedDiagnosis;
    }
    
    public void setConfirmedDiagnosis(Boolean confirmedDiagnosis) {
        this.confirmedDiagnosis = confirmedDiagnosis;
    }
    
    public Prediction getPrediction() {
        return prediction;
    }
    
    public void setPrediction(Prediction prediction) {
        this.prediction = prediction;
    }
    
    public Doctor getDoctor() {
        return doctor;
    }
    
    public void setDoctor(Doctor doctor) {
        this.doctor = doctor;
    }
} 