package backend.backend.service;

import backend.backend.dto.InferenceRequest;
import backend.backend.dto.InferenceResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.client.RestClientException;

@Service
public class MLInferenceService {
    
    @Value("${ml.service.url:http://localhost:8000}")
    private String mlServiceUrl;
    
    @Autowired
    private RestTemplate restTemplate;
    
    public InferenceResponse predict(InferenceRequest request) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<InferenceRequest> entity = new HttpEntity<>(request, headers);
            
            ResponseEntity<InferenceResponse> response = restTemplate.exchange(
                mlServiceUrl + "/predict",
                HttpMethod.POST,
                entity,
                InferenceResponse.class
            );
            
            return response.getBody();
        } catch (RestClientException e) {
            throw new RuntimeException("Errore nella comunicazione con il servizio ML: " + e.getMessage(), e);
        }
    }
} 