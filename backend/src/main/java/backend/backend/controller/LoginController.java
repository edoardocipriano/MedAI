package backend.backend.controller;

import backend.backend.entity.Doctor;
import backend.backend.repository.DoctorRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class LoginController {
    
    @Autowired
    private DoctorRepository doctorRepository;
    
    @Autowired
    private PasswordEncoder passwordEncoder;
    
    @GetMapping("/login")
    public String showLoginPage() {
        return "login";
    }
    
    @GetMapping("/register")
    public String showRegisterPage(Model model) {
        model.addAttribute("doctor", new Doctor());
        return "register";
    }
    
    @PostMapping("/register")
    public String registerDoctor(@RequestParam String firstName,
                                @RequestParam String lastName,
                                @RequestParam String specialization,
                                @RequestParam String email,
                                @RequestParam String password,
                                Model model) {
        try {
            if (doctorRepository.existsByEmail(email)) {
                model.addAttribute("error", "Email già registrata");
                return "register";
            }
            
            Doctor doctor = new Doctor();
            doctor.setFirstName(firstName);
            doctor.setLastName(lastName);
            doctor.setSpecialization(specialization);
            doctor.setEmail(email);
            doctor.setPassword(passwordEncoder.encode(password));
            
            doctorRepository.save(doctor);
            
            return "redirect:/login?registered";
        } catch (Exception e) {
            model.addAttribute("error", "Errore durante la registrazione: " + e.getMessage());
            return "register";
        }
    }
} 