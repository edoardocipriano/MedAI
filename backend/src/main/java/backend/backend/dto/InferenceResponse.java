package backend.backend.dto;

public class InferenceResponse {
    private String diabetes;
    private double probability;
    
    // Constructors
    public InferenceResponse() {}
    
    public InferenceResponse(String diabetes, double probability) {
        this.diabetes = diabetes;
        this.probability = probability;
    }
    
    // Getters and Setters
    public String getDiabetes() {
        return diabetes;
    }
    
    public void setDiabetes(String diabetes) {
        this.diabetes = diabetes;
    }
    
    public double getProbability() {
        return probability;
    }
    
    public void setProbability(double probability) {
        this.probability = probability;
    }
} 