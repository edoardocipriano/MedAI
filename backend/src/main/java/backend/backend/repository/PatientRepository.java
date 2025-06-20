package backend.backend.repository;

import backend.backend.entity.Patient;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;

@Repository
public interface PatientRepository extends JpaRepository<Patient, Long> {
    Optional<Patient> findByFiscalCode(String fiscalCode);
    boolean existsByFiscalCode(String fiscalCode);
    List<Patient> findByDoctorIdOrderByLastNameAscFirstNameAsc(Long doctorId);
} 