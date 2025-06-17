package backend.backend.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "medical_records")
public class MedicalRecord {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false)
    private LocalDateTime createdAt;
    
    @Column(columnDefinition = "TEXT")
    private String symptoms;
    
    @Column(columnDefinition = "TEXT")
    private String bloodTestResults;
    
    @Column(columnDefinition = "TEXT")
    private String additionalNotes;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "patient_id", nullable = false)
    private Patient patient;
    
    @OneToOne(mappedBy = "medicalRecord", fetch = FetchType.LAZY)
    private Prediction prediction;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }
    
    // Constructors
    public MedicalRecord() {}
    
    public MedicalRecord(Patient patient, String symptoms, String bloodTestResults, 
                        String additionalNotes) {
        this.patient = patient;
        this.symptoms = symptoms;
        this.bloodTestResults = bloodTestResults;
        this.additionalNotes = additionalNotes;
    }
    
    // Getters and Setters
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
    }
    
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
    
    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
    
    public String getSymptoms() {
        return symptoms;
    }
    
    public void setSymptoms(String symptoms) {
        this.symptoms = symptoms;
    }
    
    public String getBloodTestResults() {
        return bloodTestResults;
    }
    
    public void setBloodTestResults(String bloodTestResults) {
        this.bloodTestResults = bloodTestResults;
    }
    
    public String getAdditionalNotes() {
        return additionalNotes;
    }
    
    public void setAdditionalNotes(String additionalNotes) {
        this.additionalNotes = additionalNotes;
    }
    
    public Patient getPatient() {
        return patient;
    }
    
    public void setPatient(Patient patient) {
        this.patient = patient;
    }
    
    public Prediction getPrediction() {
        return prediction;
    }
    
    public void setPrediction(Prediction prediction) {
        this.prediction = prediction;
    }
} 