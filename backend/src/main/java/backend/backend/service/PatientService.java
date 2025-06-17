package backend.backend.service;

import backend.backend.dto.InferenceRequest;
import backend.backend.dto.InferenceResponse;
import backend.backend.entity.*;
import backend.backend.repository.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.annotation.Isolation;
import java.time.LocalDate;
import java.time.Period;
import java.util.List;
import java.util.Optional;

@Service
@Transactional
public class PatientService {
    
    @Autowired
    private PatientRepository patientRepository;
    
    @Autowired
    private MedicalRecordRepository medicalRecordRepository;
    
    @Autowired
    private PredictionRepository predictionRepository;
    
    @Autowired
    private ReviewRepository reviewRepository;
    
    @Autowired
    private MLInferenceService mlInferenceService;
    
    public List<Patient> getAllPatients() {
        return patientRepository.findAll();
    }
    
    public List<Patient> getPatientsByDoctor(Long doctorId) {
        return patientRepository.findByDoctorIdOrderByLastNameAscFirstNameAsc(doctorId);
    }
    
    public Optional<Patient> getPatientById(Long id) {
        return patientRepository.findById(id);
    }
    
    public Patient savePatient(Patient patient, Doctor doctor) {
        if (patientRepository.existsByFiscalCode(patient.getFiscalCode())) {
            throw new RuntimeException("Esiste già un paziente con questo codice fiscale");
        }
        patient.setDoctor(doctor);
        return patientRepository.save(patient);
    }
    
    public Patient updatePatient(Patient patient) {
        return patientRepository.save(patient);
    }
    
    public void deletePatient(Long id) {
        patientRepository.deleteById(id);
    }
    
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public MedicalRecord createMedicalRecordWithPrediction(Patient patient, MedicalRecord medicalRecord, 
                                                           InferenceRequest inferenceData) {
        try {
            // Ricarica il paziente per assicurarsi che sia nella sessione corrente
            Patient managedPatient = patientRepository.findById(patient.getId())
                    .orElseThrow(() -> new RuntimeException("Paziente non trovato"));
            
            // Crea nuovo record medico (non riutilizzare l'oggetto passato)
            MedicalRecord newRecord = new MedicalRecord();
            newRecord.setPatient(managedPatient);
            newRecord.setSymptoms(medicalRecord.getSymptoms());
            newRecord.setBloodTestResults(medicalRecord.getBloodTestResults());
            newRecord.setAdditionalNotes(medicalRecord.getAdditionalNotes());
            
            // Salva il record medico
            MedicalRecord savedRecord = medicalRecordRepository.save(newRecord);
            medicalRecordRepository.flush(); // Forza il flush immediato
            
            // Imposta l'età del paziente nei dati di inferenza
            int age = Period.between(managedPatient.getDateOfBirth(), LocalDate.now()).getYears();
            inferenceData.setAge(age);
            inferenceData.setGender(managedPatient.getGender());
            
            // Richiama il servizio ML per la predizione
            InferenceResponse mlResponse = mlInferenceService.predict(inferenceData);
            
            // Crea e salva la predizione
            Prediction prediction = new Prediction();
            prediction.setMedicalRecord(savedRecord);
            prediction.setPredictedDisease(mlResponse.getDiabetes().equals("Yes") ? "Diabete" : "Nessun Diabete");
            prediction.setConfidenceScore(mlResponse.getProbability());
            predictionRepository.save(prediction);
            predictionRepository.flush(); // Forza il flush immediato
            
            // Ritorna il record senza ricaricare
            return savedRecord;
            
        } catch (Exception e) {
            throw new RuntimeException("Errore durante il salvataggio: " + e.getMessage(), e);
        }
    }
    
    public List<MedicalRecord> getPatientMedicalRecords(Long patientId) {
        return medicalRecordRepository.findByPatientIdOrderByCreatedAtDesc(patientId);
    }
    
    public Optional<Prediction> getPredictionById(Long id) {
        return predictionRepository.findByIdWithRelations(id);
    }
    
    public List<Prediction> getPendingPredictions() {
        return predictionRepository.findByIsConfirmedByDoctorFalseOrderByPredictionDateDesc();
    }
    
    public List<Prediction> getPendingPredictionsByDoctor(Long doctorId) {
        return predictionRepository.findPendingPredictionsByDoctorId(doctorId);
    }
    
    public Review createReview(Prediction prediction, Doctor doctor, String notes, Boolean confirmed) {
        Review review = new Review(prediction, doctor, notes, confirmed);
        Review savedReview = reviewRepository.save(review);
        
        prediction.setIsConfirmedByDoctor(true);
        prediction.setReview(savedReview);
        predictionRepository.save(prediction);
        
        return savedReview;
    }
    
    public List<Review> getDoctorReviews(Long doctorId) {
        return reviewRepository.findByDoctorIdOrderByReviewDateDesc(doctorId);
    }
} 