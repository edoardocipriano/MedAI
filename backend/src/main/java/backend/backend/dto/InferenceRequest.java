package backend.backend.dto;

public class InferenceRequest {
    private double age;
    private String gender;
    private int hypertension;
    private int heart_disease;
    private String smoking_history;
    private double bmi;
    private double hba1c_level;
    private double blood_glucose_level;
    
    // Constructors
    public InferenceRequest() {}
    
    // Getters and Setters
    public double getAge() {
        return age;
    }
    
    public void setAge(double age) {
        this.age = age;
    }
    
    public String getGender() {
        return gender;
    }
    
    public void setGender(String gender) {
        this.gender = gender;
    }
    
    public int getHypertension() {
        return hypertension;
    }
    
    public void setHypertension(int hypertension) {
        this.hypertension = hypertension;
    }
    
    public int getHeart_disease() {
        return heart_disease;
    }
    
    public void setHeart_disease(int heart_disease) {
        this.heart_disease = heart_disease;
    }
    
    public String getSmoking_history() {
        return smoking_history;
    }
    
    public void setSmoking_history(String smoking_history) {
        this.smoking_history = smoking_history;
    }
    
    public double getBmi() {
        return bmi;
    }
    
    public void setBmi(double bmi) {
        this.bmi = bmi;
    }
    
    public double getHba1c_level() {
        return hba1c_level;
    }
    
    public void setHba1c_level(double hba1c_level) {
        this.hba1c_level = hba1c_level;
    }
    
    public double getBlood_glucose_level() {
        return blood_glucose_level;
    }
    
    public void setBlood_glucose_level(double blood_glucose_level) {
        this.blood_glucose_level = blood_glucose_level;
    }
} 